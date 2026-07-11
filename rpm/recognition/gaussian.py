import jax
import jax.numpy as np
import jax.random as jr
import flax.linen as nn

from jax import Array, vmap
from rpm.recognition.distmaps import DistMap
from rpm.prior.gaussian import GaussianNatParam
from typing import Callable



INITIALIZER = jax.nn.initializers.variance_scaling(
    scale=0.1, mode='fan_in', distribution='truncated_normal'
)

class GaussianRecognition(nn.Module):
    network: nn.Module
    dist_map: DistMap
    kernel_init: Callable = INITIALIZER
    bias_init: Callable = jax.nn.initializers.zeros
    constant_cov: bool = (
        False  # if True, recognition covariance is constant across all data
    )

    @nn.compact
    def __call__(self, x: Array) -> GaussianNatParam:
        x = self.network(x)
        if self.constant_cov:
            mean_dim = self.dist_map.latent_dim
            cov_dim = self.dist_map.input_dim - mean_dim
            cov_flat = self.variable(
                'params', 'cov', np.zeros, (cov_dim,)
            ) 
            x = nn.Dense(mean_dim, kernel_init=INITIALIZER)(x)
            x = np.concatenate((x, cov_flat.value))
        else:
            x = nn.Dense(
                self.dist_map.input_dim,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init
            )(x)

        return self.dist_map(x)


def apply(rec, params, data):
    """
    
    Parameters
    ----------
    rec:     The recognition neural network
    params:  The current parameters to use in the NN
    data:    (B, T, *_)  observations of batch B independent timeseries of length T, and of unknown complexity

    Returns
    -------
    factors: Tuple (means, covariances), where 
                - means has shape (B, T, latent_dim)
                - covariances has shape (B, T, latent_dim, latent_dim).
    """
    factors = vmap(vmap(lambda y: rec.apply(params, y)))(data)
    return factors