"""Implementations of algorithms for continuous control."""
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import MLP, Ensemble, StateActionValue, Multiplier, StateValue, DDPM, FourierFeatures, cosine_beta_schedule, ddpm_sampler, MLPResNet, get_weight_decay_mask, vp_beta_schedule

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({'params': critic_params}, observations, actions)
    q_values = q_values.min(axis=0)
    return q_values

@partial(jax.jit, static_argnames=('value_fn'))
def compute_v(value_fn, value_params, observations):
    v_values = value_fn({'params': value_params}, observations)
    return v_values

@partial(jax.jit, static_argnames=('multiplier_alpha_fn'))
def compute_a(multiplier_alpha_fn, multiplier_alpha_params, observations):
    alpha = multiplier_alpha_fn({"params": multiplier_alpha_params}, observations)
    return alpha

@partial(jax.jit, static_argnames=('multiplier_beta_fn'))
def compute_b(multiplier_beta_fn, multiplier_beta_params, observations):
    beta = multiplier_beta_fn({"params": multiplier_beta_params}, observations)
    return beta


@jax.jit
def compute_weight1(action_values,values):
    weight = -(action_values-values)**2 #eta=1
    # weight = action_values-values # eta=-1
    return weight

@jax.jit
def compute_weight2(action_values,alpha,beta):
    weight = jnp.exp(-beta*action_values-1) 
    return weight


class ALIGNIQLLearner(Agent):
    score_model: TrainState
    target_score_model: TrainState
    critic: TrainState
    target_critic: TrainState
    multiplier_alpha: TrainState
    multiplier_beta: TrainState
    value: TrainState
    discount: float
    tau: float
    actor_tau: float
    critic_hyperparam: float
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int #How many samples per observation
    M: int = struct.field(pytree_node=False) #How many repeat last steps
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    policy_temperature: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray
    use_multiplier: bool = struct.field(pytree_node=False)
    

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_hyperparam: float = 0.7,
        ddpm_temperature: float = 1.0,
        num_qs: int = 2,
        actor_num_blocks: int = 3,
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_layer_norm: bool = True,
        value_layer_norm: bool = True,
        policy_temperature: float = 3.0,
        T: int = 5,
        time_dim: int = 128,
        N: int = 64,
        M: int = 0,
        clip_sampler: bool = True,
        beta_schedule: str = 'vp',
        decay_steps: Optional[int] = int(3e6),
        use_multiplier: bool = True,
        critic_dropout: Optional[float] = None,
    ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, multiplier_alpha_key,multiplier_beta_key = jax.random.split(rng, 6)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        preprocess_time_cls = partial(FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)

        cond_model_cls = partial(MLP,
                                hidden_dims=(time_dim * 2, time_dim * 2),
                                activations=nn.swish,
                                activate_final=False)
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        base_model_cls = partial(MLPResNet,
                                    use_layer_norm=actor_layer_norm,
                                    num_blocks=actor_num_blocks,
                                    dropout_rate=actor_dropout_rate,
                                    out_dim=action_dim,
                                    activations=nn.swish)
        
        actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                            cond_encoder_cls=cond_model_cls,
                            reverse_encoder_cls=base_model_cls)

        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        actor_params = actor_def.init(actor_key, observations, actions,
                                        time)['params']

        score_model = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=optax.adamw(learning_rate=actor_lr))
        
        target_score_model = TrainState.create(apply_fn=actor_def.apply,
                                               params=actor_params,
                                               tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))

        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, 
                                  activate_final=True,
                                  dropout_rate=critic_dropout)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic_optimiser = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=critic_optimiser
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, 
                                 use_layer_norm=value_layer_norm,
                                 activate_final=True,
                                 dropout_rate=critic_dropout)
        value_def = StateValue(base_cls=value_base_cls)
        value_params = value_def.init(value_key, observations)["params"]
        value_optimiser = optax.adam(learning_rate=value_lr)

        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=value_optimiser)
        
        multiplier_alpha_cls = partial(MLP, hidden_dims=critic_hidden_dims, 
                            use_layer_norm = True,
                            activate_final=True)
        
        alpha_def = Multiplier(base_cls=multiplier_alpha_cls)
        multiplier_alpha_params = alpha_def.init(multiplier_alpha_key, observations)["params"]
        multiplier_alpha_optimiser =  optax.chain(
            optax.clip_by_global_norm(1),
            optax.adam(learning_rate=3e-5)
        )

        multiplier_alpha = TrainState.create(apply_fn=alpha_def.apply,
                                  params=multiplier_alpha_params,
                                  tx=multiplier_alpha_optimiser)
        
        multiplier_beta_cls = partial(MLP, hidden_dims=critic_hidden_dims, 
                            use_layer_norm = True,
                            activate_final=True)
        beta_def = Multiplier(base_cls=multiplier_beta_cls)
        multiplier_beta_params = beta_def.init(multiplier_beta_key, observations)["params"]
        multiplier_beta_optimiser = optax.chain(
            optax.clip_by_global_norm(1),
            optax.adam(learning_rate=3e-5)
        )

        multiplier_beta = TrainState.create(apply_fn=beta_def.apply,
                                  params=multiplier_beta_params,
                                  tx=multiplier_beta_optimiser)
        
        

        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(T)])

        return cls(
            actor=None,
            score_model=score_model,
            target_score_model=target_score_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            multiplier_alpha = multiplier_alpha,
            multiplier_beta = multiplier_beta,
            tau=tau,
            discount=discount,
            rng=rng,
            betas=betas,
            alpha_hats=alpha_hat,
            act_dim=action_dim,
            T=T,
            N=N,
            M=M,
            alphas=alphas,
            ddpm_temperature=ddpm_temperature,
            actor_tau=actor_tau,
            critic_hyperparam=critic_hyperparam,
            clip_sampler=clip_sampler,
            policy_temperature=policy_temperature,
            use_multiplier = use_multiplier,
        )
    
    def update_multiplier(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = qs.min(axis=0)

        v = agent.value.apply_fn({"params": agent.value.params}, batch["observations"])
        b = agent.multiplier_beta.apply_fn({"params": agent.multiplier_beta.params}, batch["observations"])
        a = agent.multiplier_alpha.apply_fn({"params": agent.multiplier_alpha.params}, batch["observations"])
        
        #  multiplier alpha loss
        def multiplier_alpha_loss_fn(multiplier_alpha_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            # v = agent.value.apply_fn({"params": value_params}, batch["observations"])
            a = agent.multiplier_alpha.apply_fn({"params": multiplier_alpha_params}, batch["observations"])


            loss_1 = jnp.exp(-a-b*q-1)
            loss_2 = a
            multiplier_alpha_loss = loss_1.mean()+loss_2.mean()

            return multiplier_alpha_loss, {"multiplier_alpha_loss": multiplier_alpha_loss, 
                                           "alpha": a.mean(),"weight":loss_1.mean()
                                           }
        
        #  multiplier beta loss
        def multiplier_beta_loss_fn(multiplier_beta_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            # v = agent.value.apply_fn({"params": value_params}, batch["observations"])
           
            b = agent.multiplier_beta.apply_fn({"params": multiplier_beta_params}, batch["observations"])
            beta_loss_1 = jnp.exp(-a-b*q-1)
            beta_loss_2 = v*b
            multiplier_beta_loss = beta_loss_1.mean()+beta_loss_2.mean()

            return multiplier_beta_loss, {"multiplier_beta_loss": multiplier_beta_loss,"beta":b.mean()}
        
        alpha_grads, alpha_info = jax.grad(multiplier_alpha_loss_fn, has_aux=True)(agent.multiplier_alpha.params)
        multiplier_alpha = agent.multiplier_alpha.apply_gradients(grads=alpha_grads)
        agent = agent.replace(multiplier_alpha=multiplier_alpha)
       
        beta_grads, beta_info = jax.grad(multiplier_beta_loss_fn, has_aux=True)(agent.multiplier_beta.params)
        multiplier_beta = agent.multiplier_beta.apply_gradients(grads=beta_grads)
        agent = agent.replace(multiplier_beta=multiplier_beta)
        
        return agent, dict(alpha_info,**beta_info)
    
    
    def update_v(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = qs.min(axis=0)

        def value_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])
            value_loss = expectile_loss(q - v, agent.critic_hyperparam).mean()

            return value_loss, {"value_loss": value_loss, "v": v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=grads)
        agent = agent.replace(value=value)
        return agent, info
    
    def update_q(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["next_observations"]
        )

        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_v

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"]
            )
            critic_loss = ((qs - target_q) ** 2).mean()

            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)
        agent = agent.replace(critic=critic)

        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)
        new_agent = agent.replace(critic=critic, target_critic=target_critic)
        return new_agent, info

    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)
        time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, agent.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(
            key, (batch['actions'].shape[0], agent.act_dim))
        
        alpha_hats = agent.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)

        def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = agent.score_model.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       noisy_actions,
                                       time,
                                       rngs={'dropout': key},
                                       training=True)
            
            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis = -1)).mean()
            return actor_loss, {'actor_loss': actor_loss}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)

        agent = agent.replace(score_model=score_model)
        target_score_params = optax.incremental_update(
            score_model.params, agent.target_score_model.params, agent.actor_tau
        )

        target_score_model = agent.target_score_model.replace(params=target_score_params)
        new_agent = agent.replace(score_model=score_model, target_score_model=target_score_model, rng=rng)

        return new_agent, info

    def eval_actions(self, observations: jnp.ndarray):
        # evaluate actions generated by diffusion-based behavior policy through compute_weight function
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis = 0).repeat(self.N, axis = 0)

        score_params = self.target_score_model.params
        actions, rng = ddpm_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        rng, _ = jax.random.split(rng, 2)
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions)
        if self.use_multiplier:
            alpha = compute_a(self.multiplier_alpha.apply_fn,self.multiplier_alpha.params,observations)
            beta = compute_b(self.multiplier_beta.apply_fn,self.multiplier_beta.params,observations)
            # weight = compute_weight(self.multiplier_beta.apply_fn,self.multiplier_beta.params,observations,actions,alpha,self.discount,qs)
            weight = compute_weight2(qs,alpha,beta)
        else:
            vs = compute_v(self.value.apply_fn,self.value.params,observations)
            weight = compute_weight1(qs,vs)
        idx = jnp.argmax(weight)
        action = actions[idx]
        new_rng = rng
        return np.array(action.squeeze()), self.replace(rng=new_rng)
    
    def sample_implicit_policy(self, observations: jnp.ndarray):
        # evaluate actions generated by diffusion-based behavior policy through compute_weight function
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis = 0).repeat(self.N, axis = 0)

        score_params = self.target_score_model.params
        actions, rng = ddpm_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        rng, key = jax.random.split(rng, 2)
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions)
        # vs = compute_v(self.value.apply_fn,self.value.params,observations)
        if self.use_multiplier:
            alpha = compute_a(self.multiplier_alpha.apply_fn,self.multiplier_alpha.params,observations)
            beta = compute_b(self.multiplier_beta.apply_fn,self.multiplier_beta.params,observations)
            # weight = compute_weight(self.multiplier_beta.apply_fn,self.multiplier_beta.params,observations,actions,alpha,self.discount,qs)
            weight = compute_weight2(qs,alpha,beta)
        else:
            vs = compute_v(self.value.apply_fn,self.value.params,observations)
            weight = compute_weight1(qs,vs)
        # print(qs)
        # print(alpha)
        # print(beta)
        # print(weight)
        # print(weight.shape)
        # print(weight/weight.sum())
        idx = jax.random.choice(key, self.N, p = weight/weight.sum())
        action = actions[idx]
        new_rng = rng
        return np.array(action.squeeze()), self.replace(rng=new_rng)
    
    @jax.jit
    def actor_update(self, batch: DatasetDict):
        new_agent = self
        new_agent, actor_info = new_agent.update_actor(batch)
        return new_agent, actor_info
    
    @jax.jit
    def critic_update(self, batch: DatasetDict):
        new_agent = self
        new_agent, critic_info = new_agent.update_v(batch)
        new_agent, value_info = new_agent.update_q(batch)

        return new_agent, {**critic_info, **value_info}
    
    @jax.jit
    def multiplier_update(self, batch: DatasetDict):
        new_agent = self
        new_agent, multiplier_info = new_agent.update_multiplier(batch)
        return new_agent, multiplier_info

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        new_agent, actor_info = new_agent.update_actor(batch)
        new_agent, value_info = new_agent.update_v(batch)
        new_agent, critic_info = new_agent.update_q(batch)
        if self.use_multiplier:           
            new_agent, multiplier_info = new_agent.update_multiplier(batch)
            return new_agent, {**actor_info, **critic_info, **value_info,**multiplier_info}
        else:
            return new_agent, {**actor_info, **critic_info, **value_info}