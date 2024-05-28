import flax.linen as nn
import jax.numpy as jnp

from jaxrl5.networks import default_init


class Multiplier(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        outputs = self.base_cls()(observations, *args, **kwargs)

        m = nn.Dense(1, kernel_init=default_init(), name="OutputVDense")(outputs)

        # return jnp.squeeze(value, -1)
        return jnp.squeeze(m, -1)


# class MultiplierBeta(nn.Module):
#     base_cls: nn.Module

#     @nn.compact
#     def __call__(
#         self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
#     ) -> jnp.ndarray:
#         inputs = jnp.concatenate([observations, actions], axis=-1)
#         outputs = self.base_cls()(inputs, *args, **kwargs)

#         beta = nn.Dense(1, kernel_init=default_init())(outputs)

#         return beta

