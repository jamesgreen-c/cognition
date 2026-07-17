import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn

from jax import vmap, Array
from rp_ssm.recognition.distmaps import DistMap
from rp_ssm.distributions import NatParam
from typing import Callable

from rp_ssm.config import Config
from rp_ssm.distributions import NatParam, LGStationaryParam, AllParams, NetworkParams


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
    def __call__(self, x: Array) -> NatParam:
        x = self.network(x)
        if self.constant_cov:
            mean_dim = self.dist_map.latent_dim
            cov_dim = self.dist_map.input_dim - mean_dim
            cov_flat = self.variable(
                'params', 'cov', jnp.zeros, (cov_dim,)
            )
            x = nn.Dense(mean_dim, kernel_init=INITIALIZER)(x)
            x = jnp.concatenate((x, cov_flat.value))
        else:
            x = nn.Dense(
                self.dist_map.input_dim,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init
            )(x)

        return self.dist_map(x)


class RPSSM:
    def __init__(
            self,
            prior: LGStationaryParam,
            recognition: list[GaussianRecognition]
    ):
        self.prior = prior
        self.recognition = recognition
        self.latent_dim = self.prior.latent_dim

    def init(
            self,
            key: Array,
            data: tuple[Array],
            config: Config
        ) -> AllParams:
        J = len(data)
        prior_key, *rec_keys = jr.split(key, J+1)

        # prior_params = self.prior.init(prior_key, data)
        # for now just init prior at whatever parameters you pass it
        prior_params = self.prior.opt_param # returns only learnable parameters
        rec_params = [enc.init(k, x[0,0]) for enc, k, x in zip(self.recognition, rec_keys, data)]
        
        params = (prior_params, *rec_params)
        return params
    
    def get_factors(self, rec_params: list[NetworkParams], data: list[Array]) -> list[NatParam]:
        # assume data is list of length J with each element being BxTxN
        outs = [vmap(vmap(lambda x: rec.apply(p, x)))(datapoint) for rec, p, datapoint in zip(self.recognition, rec_params, data)]
        # print(len(outs))
        return type(outs[0])(**{k: jnp.stack([out.params[k] for out in outs]) for k in outs[0].params.keys()})
    
    def rollout(
            self, 
            params: dict, 
            z_1: Array, 
            T: int, 
            key: Array, 
            cov_scale: int = 1,
        ) -> Array:
        """
        
        Rollout the prior distribution for T steps.
        If parameter (e.g. A or Q) is in param dict then use those values,
            else use the values from the prior distribution. 
        """ 
        A = params.get('A', self.prior.params['A'])
        # Q = params.get('Q', self.prior.params['Q'])
        Q = jnp.eye(A.shape[0]) - A @ A.T

        assert z_1.shape[0] == self.latent_dim, f"z_1 shape: {z_1.shape} not equal to prior latent_dim: {self.latent_dim}"
        
        # promote to diagonal matrix if A is a vector
        if A.ndim == 1:
            A = jnp.diag(A)
        
        latent_rollout = [z_1]
        for t in range(T-1):
            key, subkey = jr.split(key)
            z_t = (A @ latent_rollout[-1]) + jr.multivariate_normal(subkey, jnp.zeros(self.latent_dim), Q * cov_scale)
            latent_rollout.append(z_t)
        return jnp.stack(latent_rollout)  # shape: TxK

