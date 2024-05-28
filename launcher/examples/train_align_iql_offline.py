import os
import jax
import numpy as np
from absl import app, flags

from examples.states.train_diffusion_offline import call_main
from launcher.hyperparameters import set_hyperparameters

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
FLAGS = flags.FLAGS
flags.DEFINE_integer('variant', 0, 'Logging interval.')
flags.DEFINE_integer('device', 0, 'default jax device.')

os.environ["WANDB_MODE"]="offline"

def main(_):
    constant_parameters = dict(project='align_iql_offline',
                               experiment_name='align_iql',
                               max_steps=3000001, #Actor takes two steps per critic step
                               batch_size=512,
                               eval_episodes=50,
                               log_interval=1000,
                               eval_interval=250000,
                               save_video = False,
                               filter_threshold=None,
                               take_top=None,
                               online_max_steps = 0,
                               unsquash_actions=False,
                               normalize_returns=True,
                               T=5,
                               critic_hyperparam = 0.7,
                               beta_schedule = 'vp',
                               training_time_inference_params=dict(
                                N = 64,
                                clip_sampler = True,
                                M = 0,),
                               rl_config=dict(
                                   model_cls='ALIGNIQLLearner',
                                   actor_lr=3e-4,
                                   critic_lr=3e-4,
                                   value_lr=3e-4,
                                   T=5,
                                   N=64,
                                   M=0,
                                   actor_dropout_rate=0.1,
                                   actor_num_blocks=3,
                                   decay_steps=int(3e6),
                                   actor_layer_norm=True,
                                   value_layer_norm=True,
                                   actor_tau=0.001,
                                   beta_schedule='vp',
                                   use_multiplier = False,
                                   critic_dropout = None,
                               ))

    sweep_parameters = dict(
                            seed=list(range(4)),
                            # beta_schedule = ['vp','cosine','linear'],
                            # critic_hyperparam =[0.7,0.8,0.9,0.99],
                            # T = [5,10,20],
                            env_name=['walker2d-medium-v2', 'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2',  
                            'halfcheetah-medium-v2', 'halfcheetah-medium-replay-v2', 'halfcheetah-medium-expert-v2',
                            'hopper-medium-v2', 'hopper-medium-replay-v2', 'hopper-medium-expert-v2', 
                            'antmaze-umaze-v2', 'antmaze-umaze-diverse-v2', 'antmaze-medium-diverse-v2', 
                            'antmaze-medium-play-v2', 'antmaze-large-diverse-v2', 'antmaze-large-play-v2',],
                            )
    

    variants = [constant_parameters]
    name_keys = ['experiment_name', 'env_name']
    variants = set_hyperparameters(sweep_parameters, variants, name_keys)
    # print(len(variants))

    inference_sweep_parameters = dict(
                            N = [16, 64, 256],
                            clip_sampler = [True], 
                            M = [0],
                            )
    
    inference_variants = [{}]
    name_keys = []
    inference_variants = set_hyperparameters(inference_sweep_parameters, inference_variants)
    # print(inference_variants)


    filtered_variants = []
    for variant in variants:
        # variant['rl_config']['T'] = variant['T']
        # variant['rl_config']['beta_schedule'] = variant['beta_schedule']
        variant['inference_variants'] = inference_variants
        # variant['rl_config']['critic_hyperparam'] = variant['critic_hyperparam']
            
        if 'antmaze' in variant['env_name']:
            variant['rl_config']['critic_hyperparam'] = 0.9
        else:
            variant['rl_config']['critic_hyperparam'] = 0.7

        filtered_variants.append(variant)

    print(len(filtered_variants))
    variant = filtered_variants[FLAGS.variant]
    print(FLAGS.variant)
    jax.config.update("jax_default_device", jax.devices()[FLAGS.device])
    print(FLAGS.device)

    # print(variant)
    call_main(variant)


if __name__ == '__main__':
    app.run(main)
