from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn

from jax import vmap, Array

from rp_slac.recognition.distmaps import DistMap
from rp_slac.config import Config
from rp_slac.distributions import LGStationaryParam, AllParams, NetworkParams
from rp_slac.utils.kalman import parallel_filter


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

    def encode(self, x: Array) -> Array:
        """Shared observation features, before the Gaussian recognition head."""
        return self.network(x)

    @nn.compact
    def __call__(self, x: Array, return_features: bool = False):

        features = self.encode(x)
        x = features
        
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

        factor = self.dist_map(x)
        return (factor, features) if return_features else factor


class RPSSM:
    def __init__(
            self,
            prior: LGStationaryParam,
            recognition: list[GaussianRecognition],
            reward_network: GaussianRecognition | None = None,
    ):
        self.prior = prior
        self.recognition = recognition
        self.latent_dim = self.prior.latent_dim
        self.action_dim = self.prior.params['B'].shape[-1]
        self.reward_network = reward_network

    def _data(self, data):
        data = tuple(data) if isinstance(data, (tuple, list)) else (data,)

        if not data or len(data) != len(self.recognition):
            raise ValueError('Expected one observation array per recogniser.')
        if any(x.shape[:2] != data[0].shape[:2] for x in data):
            raise ValueError('Observation arrays must share batch and time axes.')
        if data[0].shape[1] < 1:
            raise ValueError('At least one observation is required.')
        
        return data

    def _actions(self, actions, batch, times):

        if actions is None:
            if self.action_dim:
                raise ValueError('Recorded actions are required for this model.')
            return jnp.zeros((batch, times - 1, 0))
        if actions.ndim == 2 and self.action_dim == 1:
            actions = actions[..., None]
        if actions.shape != (batch, times - 1, self.action_dim):
            raise ValueError('Expected actions shaped (B,T-1,action_dim).')
        
        return actions

    def get_prior(self, prior_params):
        """Update the LGSSM without treating reward-network weights as dynamics."""
        return self.prior.update({k: v for k, v in prior_params.items() if k != 'reward'})

    def init(
            self,
            key: Array,
            data: tuple[Array] | Array,
            config: Config
        ) -> AllParams:
        data = self._data(data)
        prior_key, *rec_keys = jr.split(key, len(data) + 1)

        # prior_params = self.prior.init(prior_key, data)
        # for now just init prior at whatever parameters you pass it
        prior_params = dict(self.prior.opt_param)  # returns only learnable parameters
        rec_params = [enc.init(k, x[0,0]) for enc, k, x in zip(self.recognition, rec_keys, data)]

        if self.reward_network is not None:
            inputs = jnp.zeros(2 * self.latent_dim + self.action_dim)
            prior_params['reward'] = self.reward_network.init(prior_key, inputs)

        return {"prior": prior_params, "rpm": rec_params}
    
    def get_factors(self, rec_params: list[NetworkParams], data, return_features: bool = False):
        """
        Apply the observation recognisers over batch and time.

        Parameters
        ----------
        rec_params: (J,). Observation-network Flax variable trees, one per recogniser.
        data:       (J, ...) or (...) if J=1. Observation arrays shaped (B,T,*O_j).

        Returns
        -------
        output: Observation ratios with pwm=(J,B,T,N), p=(J,B,T,N,N).
        """
        data = self._data(data)

        if len(rec_params) != len(self.recognition):
            raise ValueError('Recognition parameter count does not match the model.')
        
        # apply recognition to data and remove features if not required
        outputs = [
            vmap(vmap(lambda x: rec.apply(p, x, return_features=return_features)))(datapoint)
            for rec, p, datapoint in zip(self.recognition, rec_params, data)
        ]
        outs = [out[0] for out in outputs] if return_features else outputs

        # stack outputted statistics into a single NatParam for all factors
        factors = type(outs[0])(
            **{k: jnp.stack([out.params[k] for out in outs]) for k in outs[0].params}
        )

        if factors.latent_dim != self.latent_dim:
            raise ValueError('Recognition and prior latent dimensions must match.')
        return (factors, outputs[0][1]) if return_features else factors

    def get_features(self, rec_params, observations: Array) -> Array:
        """Encode (B,T,...) frames for the existing shared-feature SAC actor."""
        # TODO this should loop over rec: (J,), data: (J,)
        rec = self.recognition[0]
        _apply = lambda frame: rec.apply(rec_params[0], frame, method=rec.encode)
        return vmap(vmap(_apply))(observations)

    def predict_rewards(self, prior_params, latents: Array, actions: Array | None):
        """Gaussian reward predictions from (z_t, a_t, z_{t+1})."""

        if self.reward_network is None:
            raise ValueError('This RPSSM has no reward predictor.')
        
        # reward predictor requires actions and transitions
        actions = self._actions(actions, *latents.shape[:2])
        inputs = jnp.concatenate((latents[:, :-1], actions, latents[:, 1:]), axis=-1)
        _apply = lambda inp: self.reward_network.apply(prior_params['reward'], inp)
        return vmap(vmap(_apply))(inputs)

    def get_causal_distribution(self, prior_params, factors_nat, actions):
        """RPSSM equivalent of RAMP's forward message pass.

        factors_nat contains observation factors with pwm (J,B,T,D) and
        p (J,B,T,D,D). Return a batched LGFilterDistParam; inference and
        sampling use observations/actions only, never reward targets.
        """
        prior = self.get_prior(prior_params)
        factors = factors_nat.sum(axis=0).dist_param
        actions = self._actions(actions, *factors.params['mean'].shape[:2])
        return vmap(lambda f, a: parallel_filter(prior, f, a))(factors, actions)

    
#     def initial_distribution(self, params, data):
#         """
#         Get filter distribution at T=0 given the initial prior and observation.

#         Parameters
#         ----------
#         params:  tuple (prior_params, *rec_params)
#         data:    (B, *D) T=0 observation for each recognition factor
#         """
#         if isinstance(params, dict):
#             prior_params, rec_params = params['prior'], params['rpm']
#         else:
#             prior_params, *rec_params = params
#         prior = self.get_prior(prior_params)

#         outs = [vmap(lambda x: rec.apply(p, x))(datapoint) for rec, p, datapoint in zip(self.recognition, rec_params, data)]
#         factors_nat = type(outs[0])(**{k: jnp.stack([out.params[k] for out in outs]) for k in outs[0].params.keys()}) # JxBxK
#         factors = AllParam(factors_nat.sum(axis=0)) # BxK

#         return initial_filter_distribution(prior, factors.dist_param, self.latent_dim)

#     def filter(self, params, posterior, action, data):
#         """ 
#         Get filter distribution for next state given an observation 
        
#         Parameters
#         ----------
#         params:     tuple (prior_params, *rec_params)
#         posterior:  q(z_{t-1} | x_0:t-1, a_0:t-2)
#         action:     a_{t-1}, shaped (B, action_dim)
#         data:       (B, *D) T=t observation to get factor from
#         """
#         prior = self.get_prior(params["prior"])
#         rec_params = params["rpm"]

#         outs = [vmap(lambda x: rec.apply(p, x))(datapoint) for rec, p, datapoint in zip(self.recognition, rec_params, data)]
#         factors_nat = type(outs[0])(**{k: jnp.stack([out.params[k] for out in outs]) for k in outs[0].params.keys()}) # JxBxK
#         factors = AllParam(factors_nat.sum(axis=0)) # BxK

#         return next_filter_distribution(prior, posterior, action, factors.dist_param, self.latent_dim)

#     def rollout(
#             self, 
#             params: dict, 
#             z_1: Array, 
#             T: int, 
#             key: Array, 
#             cov_scale: int = 1,
#         ) -> Array:
#         """
        
#         Rollout the prior distribution for T steps.
#         If parameter (e.g. A or Q) is in param dict then use those values,
#             else use the values from the prior distribution. 
#         """ 
#         A = params.get('A', self.prior.params['A'])
#         # Q = params.get('Q', self.prior.params['Q'])
#         Q = jnp.eye(A.shape[0]) - A @ A.T

#         assert z_1.shape[0] == self.latent_dim, f"z_1 shape: {z_1.shape} not equal to prior latent_dim: {self.latent_dim}"
        
#         # promote to diagonal matrix if A is a vector
#         if A.ndim == 1:
#             A = jnp.diag(A)
        
#         latent_rollout = [z_1]
#         for t in range(T-1):
#             key, subkey = jr.split(key)
#             z_t = (A @ latent_rollout[-1]) + jr.multivariate_normal(subkey, jnp.zeros(self.latent_dim), Q * cov_scale)
#             latent_rollout.append(z_t)
#         return jnp.stack(latent_rollout)  # shape: TxK


# def initial_filter_distribution(prior, factors, latent_dim):
#     """
#     Get the initial filter distribution given that:

#         p(z_0) = N(m1, Q1).
#         p(z_0 | x_0) = N(f.mean, f.cov)
#     """
#     # extract params and stats
#     m1 = prior.params["m1"]
#     Q1 = prior.params["Q1"]
#     I = jnp.eye(latent_dim)

#     factor_mean = factors.params["mean"]
#     factor_cov = factors.params["cov"]
#     factor_precision = jnp.linalg.solve(factor_cov, I)

#     # prior means and covs
#     batch_size = factor_mean.shape[0]
#     prior_mean = jnp.broadcast_to(m1, (batch_size, latent_dim))
#     prior_cov = jnp.broadcast_to(Q1, (batch_size, latent_dim, latent_dim))
#     prior_precision = jnp.linalg.solve(prior_cov, I)

#     # get posterior stats
#     posterior_precision = prior_precision + factor_precision
#     posterior_cov = jnp.linalg.solve(posterior_precision, I)

#     prior_precision_mean = jnp.einsum("bij,bj->bi", prior_precision, prior_mean)
#     factor_precision_mean = jnp.einsum("bij,bj->bi", factor_precision, factor_mean)
#     posterior_mean = jnp.einsum("bij,bj->bi", posterior_cov, prior_precision_mean + factor_precision_mean)

#     # enforce symmetry
#     posterior_cov = 0.5 * (posterior_cov + jnp.swapaxes(posterior_cov, -1, -2))

#     return GaussianDistParam(mean=posterior_mean, cov=posterior_cov)


# def next_filter_distribution(prior, posterior, action, factors, latent_dim):
#     """
#     Get the next filtering distribution given:
#         q(z_{t-1} | x_{0:t-1}) = N(m_{t-1}, P_{t-1})
#         p(z_t | x_{0:t-1}) = N(A m_{t-1} + b, A P_{t-1} A.T + Q)
#         f(z_t | x_t) = N(f.mean, f.cov)
#     """

#     # extract required params and stats
#     A = prior.params["A"]
#     B = prior.params["B"]
#     Q = prior.params["Q"]
#     b = prior.params["b"]
#     I = jnp.eye(latent_dim)

#     if A.ndim == 1:
#         A = jnp.diag(A)

#     factor_mean = factors.params["mean"]
#     factor_cov = factors.params["cov"]
#     factor_precision = jnp.linalg.solve(factor_cov, I)

#     previous_mean = posterior.params["mean"]
#     previous_cov = posterior.params["cov"]

#     # Gaussian state dynamics
#     predicted_mean = previous_mean @ A.T + action @ B.T + b
#     predicted_cov = jnp.einsum("ij,bjk,lk->bil", A, previous_cov, A) + Q
#     predicted_cov = 0.5 * (predicted_cov + jnp.swapaxes(predicted_cov, -1, -2))
#     predicted_precision = jnp.linalg.solve(predicted_cov, I)

#     # get posterior stats
#     posterior_precision = predicted_precision + factor_precision
#     posterior_cov = jnp.linalg.solve(posterior_precision, I)

#     predicted_eta = jnp.einsum("bij,bj->bi", predicted_precision, predicted_mean)
#     factor_eta = jnp.einsum("bij,bj->bi", factor_precision, factor_mean)
#     posterior_mean = jnp.einsum("bij,bj->bi", posterior_cov, predicted_eta + factor_eta)

#     # enforce symmetry
#     posterior_cov = 0.5 * (posterior_cov + jnp.swapaxes(posterior_cov, -1, -2))

#     return GaussianDistParam(mean=posterior_mean, cov=posterior_cov)
