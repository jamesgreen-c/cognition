import optax
import jax.numpy as jnp

from typing import Any

from jax import vmap, Array
from jax.scipy.special import logsumexp
from jax.lax import stop_gradient as stopgrad

from rp_slac.config import Config
from rp_slac.utils.math import inv_quad_form
from rp_slac.distributions import AllParam, LGChainDistParam, AllParams
from rp_slac.recognition.rpm import RPSSM

from dynamax.linear_gaussian_ssm import parallel_lgssm_smoother
from dynamax.linear_gaussian_ssm.inference import make_lgssm_params


class ConstrainedIVFreeEnergy:
    def __init__(self, model: RPSSM):
        self.model = model

    def init(
            self,
            key: Array,
            data: tuple[Array],
            config: Config
        ) -> tuple[dict, dict[optax.OptState], dict[optax.GradientTransformation]]:
        observations, *_ = data

        self.num_timesteps = config.sequence_length + 1
        self.batch_size = config.batch_size
        self.num_buffers = config.num_buffers
        self.num_factors = 1

        params = self.model.init(key, (observations,), config)
        opts = {"prior": config.prior.build(), "rpm": config.recognition.build()}
        opt_states = {name: opts[name].init(params[name]) for name in opts}
        return params, opt_states, opts

    def loss(self, params: dict, observations: Array, actions: Array, beta: float, em: bool) -> tuple[float, Any]:
        
        prior = self.model.prior.update(params["prior"])
        # prior = stopgrad(self.model.prior.update(prior_params))

        ### E-step
        prior_chain, factors_nat, posterior = self.get_posterior(prior, params["rpm"], actions, observations)
        if em: posterior = stopgrad(posterior)

        ### M-step
        kl_qf, log_Gamma, kl_qp = self.get_loss_terms(prior_chain, factors_nat, posterior)

        Z = self.batch_size * self.num_buffers * self.num_timesteps * self.num_factors 
        loss = -(log_Gamma - kl_qf - beta * kl_qp) / Z

        aux = {
            'posterior': posterior,
            'factors_nat': factors_nat,
            'kl_qp': kl_qp / Z,
            'kl_qf': kl_qf / Z,
            'log_Gamma': log_Gamma / Z
        }

        return loss, aux
    
    def get_posterior(self, prior, rec_params, actions, observations):
        factors_nat = self.model.get_factors(rec_params, (observations,))                  # JxBxTxK
        factors_tot = AllParam(factors_nat.sum(axis=0))                                    # BxTxK

        prior_chains = vmap(lambda _acts: prior.to_chain(_acts))(actions)                  # TxK
        posterior = vmap(
            lambda f, _acts: parallel_smoother(prior, f, _acts, self.model.latent_dim)
        )(factors_tot.dist_param, actions)                                                 # BxTxK

        return prior_chains, factors_nat, posterior
        #TODO: implement flexible_vmap function for factors

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


def parallel_smoother(prior, factors, actions, latent_dim):

    C = jnp.eye(latent_dim)
    d = jnp.zeros(latent_dim) # TODO: check if C,d are automatically initialized to these values
    dynamics_bias = actions @ prior.params['B'].T + prior.params['b']

    lgssm_params = make_lgssm_params(
        prior.params['m1'],
        prior.params['Q1'],
        prior.params['A'],
        prior.params['Q'],
        C,
        factors.params['cov'],
        dynamics_bias=dynamics_bias,
        emissions_bias=d
    )
    smoother_out = parallel_lgssm_smoother(lgssm_params, factors.params['mean'])._asdict()

    # in rare cases the smoothed covariances have min. evalues ~-1e-6, so add a correction to be safe (checked that this has no effect on experiments that were already stable)
    smoother_out['smoothed_covariances'] += 1e-5 * jnp.eye(latent_dim)
    smoother_out['filtered_covariances'] += 1e-5 * jnp.eye(latent_dim)
    
    filtered_cov = smoother_out['filtered_covariances']
    smoothed_cov = smoother_out['smoothed_covariances']
    A, Q = prior.params['A'], prior.params['Q']
    cross_covs = vmap(
        lambda S, F: inv_quad_form(Q + A @ F @ A.T, S.T, A @ F)
    )(smoothed_cov[1:], filtered_cov[:-1])

    posterior = LGChainDistParam(
        means=smoother_out['smoothed_means'],
        covs=smoothed_cov,
        cross_covs=cross_covs
    )
    return posterior
