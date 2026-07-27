import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn

from jax import vmap, Array
from rp_ssm.recognition.distmaps import DistMap
from rp_ssm.distributions import NatParam, AllParam, GaussianDistParam
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

    def initial_distribution(self, params, data):
        """
        Get filter distribution at T=0 given the initial prior and observation.

        Parameters
        ----------
        params:  tuple (prior_params, *rec_params)
        data:    (B, *D) T=0 observation for each recognition factor
        """
        prior_params, *rec_params = params
        prior = self.prior.update(prior_params)

        outs = [vmap(lambda x: rec.apply(p, x))(datapoint) for rec, p, datapoint in zip(self.recognition, rec_params, data)]
        factors_nat = type(outs[0])(**{k: jnp.stack([out.params[k] for out in outs]) for k in outs[0].params.keys()}) # JxBxK
        factors = AllParam(factors_nat.sum(axis=0)) # BxK

        return initial_filter_distribution(prior, factors.dist_param, self.latent_dim)

    def filter(self, params, posterior, data):
        """ 
        Get filter distribution for next state given an observation 
        
        Parameters
        ----------
        params:  tuple (prior_params, *rec_params)
        state:   (B, K) T=t-1 posterior to apply filter to
        data:    (B, *D) T=t observation to get factor from
        """
        prior_params, *rec_params = params
        prior = self.prior.update(prior_params)

        outs = [vmap(lambda x: rec.apply(p, x))(datapoint) for rec, p, datapoint in zip(self.recognition, rec_params, data)]
        factors_nat = type(outs[0])(**{k: jnp.stack([out.params[k] for out in outs]) for k in outs[0].params.keys()}) # JxBxK
        factors = AllParam(factors_nat.sum(axis=0)) # BxK

        return next_filter_distribution(prior, posterior, factors.dist_param, self.latent_dim)

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


def initial_filter_distribution(prior, factors, latent_dim):
    """
    Get the initial filter distribution given that:

        p(z_0) = N(m1, Q1).
        p(z_0 | x_0) = N(f.mean, f.cov)
    """
    # extract params and stats
    m1 = prior.params["m1"]
    Q1 = prior.params["Q1"]
    I = jnp.eye(latent_dim)

    factor_mean = factors.params["mean"]
    factor_cov = factors.params["cov"]
    factor_precision = jnp.linalg.solve(factor_cov, I)

    # prior means and covs
    batch_size = factor_mean.shape[0]
    prior_mean = jnp.broadcast_to(m1, (batch_size, latent_dim))
    prior_cov = jnp.broadcast_to(Q1, (batch_size, latent_dim, latent_dim))
    prior_precision = jnp.linalg.solve(prior_cov, I)

    # get posterior stats
    posterior_precision = prior_precision + factor_precision
    posterior_cov = jnp.linalg.solve(posterior_precision, I)

    prior_precision_mean = jnp.einsum("bij,bj->bi", prior_precision, prior_mean)
    factor_precision_mean = jnp.einsum("bij,bj->bi", factor_precision, factor_mean)
    posterior_mean = jnp.einsum("bij,bj->bi", posterior_cov, prior_precision_mean + factor_precision_mean)

    # enforce symmetry
    posterior_cov = 0.5 * (posterior_cov + jnp.swapaxes(posterior_cov, -1, -2))

    return GaussianDistParam(mean=posterior_mean, cov=posterior_cov)


def next_filter_distribution(prior, posterior, factors, latent_dim):
    """
    Get the next filtering distribution given:
        q(z_{t-1} | x_{0:t-1}) = N(m_{t-1}, P_{t-1})
        p(z_t | x_{0:t-1}) = N(A m_{t-1} + b, A P_{t-1} A.T + Q)
        f(z_t | x_t) = N(f.mean, f.cov)
    """

    # extract required params and stats
    A = prior.params["A"]

    if A.ndim == 1:
            A = jnp.diag(A)

    Q = prior.params["Q"]
    b = prior.params["b"]
    I = jnp.eye(latent_dim)

    factor_mean = factors.params["mean"]
    factor_cov = factors.params["cov"]
    factor_precision = jnp.linalg.solve(factor_cov, I)

    previous_mean = posterior.params["mean"]
    previous_cov = posterior.params["cov"]

    # Gaussian state dynamics
    predicted_mean = previous_mean @ A.T + b
    predicted_cov = jnp.einsum("ij,bjk,lk->bil", A, previous_cov, A) + Q
    predicted_precision = jnp.linalg.solve(predicted_cov, I)

    # get posterior stats
    posterior_precision = predicted_precision + factor_precision
    posterior_cov = jnp.linalg.solve(posterior_precision, I)

    predicted_eta = jnp.einsum("bij,bj->bi", predicted_precision, predicted_mean)
    factor_eta = jnp.einsum("bij,bj->bi", factor_precision, factor_mean)
    posterior_mean = jnp.einsum("bij,bj->bi", posterior_cov, predicted_eta + factor_eta)

    # enforce symmetry
    posterior_cov = 0.5 * (posterior_cov + jnp.swapaxes(posterior_cov, -1, -2))

    return GaussianDistParam(mean=posterior_mean, cov=posterior_cov)
