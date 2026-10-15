"""RPSSM free energy and causal Gaussian trajectories for SAC.

The RPSSM objective uses its exact Gaussian smoothing posterior. Reward
supervision and SAC instead share the causal variational chain

    q(z_0 | x_0) proportional to p(z_0) f_0(z_0),
    q(z_t | z_{t-1}, x_t, a_{t-1}) proportional to
        p(z_t | z_{t-1}, a_{t-1}) f_t(z_t).

Each conditional is normalised separately. This is a Gaussian joint with
cross-time covariances, not the collection of exact Kalman-filter marginals,
and not the smoothing posterior. Future observations never enter its forward
conditionals. Rewards are targets only, and do not enter inference.
"""

import optax
import jax.numpy as jnp

from typing import Any

from jax import lax, vmap, Array
from jax.scipy.special import logsumexp

from rp_slac.config import Config
from rp_slac.distributions import AllParam
from rp_slac.recognition.rpm import RPSSM
from rp_slac.utils.kalman import parallel_smoother


class ConstrainedIVFreeEnergy:
    def __init__(self, model: RPSSM, reward_weight: float = 1.0):

        if reward_weight < 0:
            raise ValueError('reward_weight must be nonnegative.')
        
        self.model = model
        self.reward_weight = reward_weight

    def configure(self, config: Config):
        """ For easier use in RPSLAC.train_continue() """
        self.num_timesteps = config.sequence_length + 1
        self.batch_size = config.model_batch_size
        self.num_buffers = config.num_buffers
        self.num_factors = len(self.model.recognition)

    def init(
            self,
            key: Array,
            data: tuple[Array],
            config: Config
        ) -> tuple[dict, dict[optax.OptState], dict[optax.GradientTransformation]]:
        self.configure(config)

        observations, *_ = data
        params = self.model.init(key, observations, config)

        opts = {"prior": config.prior.build(), "rpm": config.recognition.build()}
        opt_states = {name: opts[name].init(params[name]) for name in opts}
        return params, opt_states, opts

    def get_posterior(self, prior_params, rec_params, actions, observations, *, factors_nat=None):

        # apply recognition to get incoming messages from data
        factors_nat, features = self.model.get_factors(rec_params, observations, return_features=True) 
        factors = AllParam(factors_nat.sum(axis=0)).dist_param      # BxTxK

        # construct prior chains
        prior = self.model.get_prior(prior_params)
        actions = self.model._actions(actions, *factors.params['mean'].shape[:2])
        prior_chains = vmap(lambda _acts: prior.to_chain(_acts))(actions)       # TxK

        # calculate posterior using Kalman smoothing
        posterior, filtered = vmap(lambda _f, _a: parallel_smoother(prior, _f, _a))(factors, actions)
        return prior_chains, factors_nat, posterior, filtered, features

    def loss(
        self, params: dict, observations: Array, actions: Array | None,
        beta: float = 1.0, em: bool = False, rewards: Array | None = None,
        discounts: Array | None = None, key: Array | None = None,
    ) -> tuple[Array, Any]:

        prior_chains, factors_nat, posterior, filtered, features = self.get_posterior(
            params["prior"], params["rpm"], actions, observations
        )

        # in EM mode hold the E-step belief fixed, but still train the messages.
        posterior = lax.cond(em, lax.stop_gradient, lambda q: q, posterior)
        filtered = lax.cond(em, lax.stop_gradient, lambda q: q, filtered)
        latents = filtered.sample(key, shape=None)

        # calculate RPSSM loss and reward loss
        kl_qf, log_Gamma, kl_qp = self.get_loss_terms(prior_chains, factors_nat, posterior)
        reward_nll, reward_mse = self.get_reward_loss_terms(
            params, latents, key, actions, rewards, discounts,
        )

        batch, times = posterior.params['means'].shape[:2]
        Z = batch * times
        model_loss = -(log_Gamma - kl_qf - beta * kl_qp) / Z
        loss = model_loss + self.reward_weight * reward_nll

        aux = {
            'posterior': posterior,
            'filter_posterior': filtered,
            'features': features,
            'factors_nat': factors_nat,
            'ramp_loss': model_loss,  # Existing trainer diagnostic name.
            'reward_nll': reward_nll,
            'reward_mse': reward_mse,
            'kl_qp': kl_qp / Z,
            'kl_qf': kl_qf / Z,
            'log_Gamma': log_Gamma / Z
        }

        return loss, aux

    def apply(self, prior_params, rec_params, observations, actions):
        """Return (LGSSM prior, causal LGChainDistParam, shared features).

        The chain has means/covs shaped (B,T,D)/(B,T,D,D) and cross_covs
        shaped (B,T-1,D,D), with Cov(z_{t+1}, z_t) orientation. Its sample
        method draws joint (B,T,D) trajectories using distributions.py.
        """
        prior = self.model.get_prior(prior_params)
        factors, features = self.model.get_factors(rec_params, observations, return_features=True)
        filtered = self.model.get_causal_distribution(prior_params, factors, actions)
        return prior, filtered, features

    def get_reward_loss_terms(
            self, params: dict, latents: Array, key = None, 
            actions: Array | None = None, rewards: Array | None = None, 
            discounts: Array | None = None,
        ):
        """
        Return reward negative log likelihood of the true reward at current predicted reward 
        distribution. Also calculate an MSE as the mean squared error of the true reward and 
        the mean predicted reward.

        Parameters
        ----------
        params:     Model parameters
        latents:    A sample of latents from the filter distribution given by the RAMPSSM
        key:        RNG
        actions:    (B,T-1,A), (B,T-1) when A=1, or None when A=0. Recorded edge actions.
        rewards:    (B,T-1) Array of rewards for given transitions.
        discounts:  (B,T-1) Array of discounts for given timesteps.

        Returns
        -------
        output:  tuple(rewards_nll, reward_mse)
        """
        reward_nll = jnp.asarray(0.0)
        reward_mse = jnp.asarray(0.0)
        batch, times = latents.shape[:2]

        if self.model.reward_network is not None:

            if rewards is None or actions is None or key is None:
                raise ValueError('Reward-supervised RAMP requires rewards, actions and a sampling key.')
            if rewards.shape != (batch, times - 1):
                raise ValueError('Expected rewards shaped (B,T-1).')
            if discounts is not None and discounts.shape != rewards.shape:
                raise ValueError('Reward and replay-flag shapes must match.')

            preds_nat = self.model.predict_rewards(params['prior'], latents, actions)
            preds = preds_nat.dist_param

            residual = rewards - preds.params["mean"][..., 0]
            nll = -vmap(vmap(lambda _pred, _reward: _pred.log_prob(_reward)))(preds, rewards[..., None])
            # nll = -preds.log_prob(rewards)
            # nll = 0.5 * (residual / std)**2 + jnp.log(std) + 0.5 * jnp.log(2.0 * jnp.pi)

            # Current replay flags identify reset edges, whose next frame is
            # a reset observation. The normal sampler excludes these edges.
            valid = jnp.ones_like(rewards) if discounts is None else (discounts > 0).astype(nll.dtype)
            denominator = jnp.maximum(valid.sum(), 1.0)
            reward_nll = (valid * nll).sum() / denominator
            reward_mse = (valid * residual**2).sum() / denominator

        return reward_nll, reward_mse

    def get_loss_terms(self, prior_chains, factors_nat, posterior):
        kl_qp = vmap(lambda qtk, ptk: qtk.kl(ptk))(posterior, prior_chains) # B

        prior_chains = prior_chains.all_param
        posterior = posterior.all_param

        kl_qf = vmap(lambda fntk: vmap(vmap(lambda qtk, ftk: qtk.kl(qtk+ftk)))(posterior.nat_param, fntk))(factors_nat) # JxBxT
        log_gammas = vmap(lambda fntk:
                          vmap(lambda qnk, fnk, pnk:
                               vmap(lambda qk, pk:
                                    vmap(lambda fk: (fk + qk).lognormalizer - (fk + pk).lognormalizer)(fnk)
                                )(qnk, pnk),
                               in_axes=(1,1,1)
                            )(posterior.nat_param, fntk, prior_chains.nat_param)
                         )(factors_nat) # JxTxBxB
        
        log_Gamma = vmap(vmap(lambda G: jnp.diag(G) - logsumexp(G, axis=1)))(log_gammas) # JxTxB
        return kl_qf.sum(), log_Gamma.sum(), kl_qp.sum()


# def parallel_smoother(prior, factors, actions):

#     latent_dim = prior.params["means"].shape[-1]
    
#     A = prior.params['A'] 
#     Q = prior.params['Q']
#     C = jnp.eye(latent_dim)
#     d = jnp.zeros(latent_dim) # TODO: check if C,d are automatically initialized to these values
#     dynamics_bias = actions @ prior.params['B'].T + prior.params['b']

#     lgssm_params = make_lgssm_params(
#         prior.params['m1'],
#         prior.params['Q1'],
#         prior.params['A'],
#         prior.params['Q'],
#         C,
#         factors.params['cov'],
#         dynamics_bias=dynamics_bias,
#         emissions_bias=d
#     )
#     smoother_out = parallel_lgssm_smoother(lgssm_params, factors.params['mean'])._asdict()

#     # in rare cases the smoothed covariances have min. evalues ~-1e-6, so add a correction to be safe (checked that this has no effect on experiments that were already stable)
#     smoother_out['smoothed_covariances'] += 1e-5 * jnp.eye(latent_dim)
#     smoother_out['filtered_covariances'] += 1e-5 * jnp.eye(latent_dim)
    
#     filtered_cov = smoother_out['filtered_covariances']
#     smoothed_cov = smoother_out['smoothed_covariances']

#     cross_covs = vmap(
#         lambda S, F: inv_quad_form(Q + A @ F @ A.T, S.T, A @ F)
#     )(smoothed_cov[1:], filtered_cov[:-1])

#     posterior = LGChainDistParam(
#         means=smoother_out['smoothed_means'],
#         covs=smoothed_cov,
#         cross_covs=cross_covs
#     )
#     filter_posterior = GaussianDistParam(
#         dist_map=None,
#         mean=smoother_out["filtered_means"],
#         cov=filtered_cov
#     )
#     return posterior, filter_posterior
