import optax
import jax.numpy as jnp

from typing import Any

from jax import vmap, Array
from jax.scipy.special import logsumexp
from jax.lax import stop_gradient as stopgrad

from rp_ssm.config import Config
from rp_ssm.utils.math import inv_quad_form
from rp_ssm.distributions import AllParam, LGChainDistParam, AllParams
from rp_ssm.recognition.rpm import RPSSM

from dynamax.linear_gaussian_ssm import parallel_lgssm_smoother
from dynamax.linear_gaussian_ssm.inference import make_lgssm_params


class ConstrainedIVFreeEnergy:
    def __init__(self, model: RPSSM):
        self.model = model

    def init(
            self,
            key: Array,
            data: list[Array],
            config: Config
        ) -> tuple[AllParams, list[optax.OptState], list[optax.GradientTransformation]]:
        self.num_timesteps = data[0].shape[1]
        self.batch_size = config.batch_size
        self.num_factors = len(data)

        params = self.model.init(key, data, config)
        opts = [config.prior_opt(config.prior_lr), *[optax.adam(lr) for lr in config.rec_lr]]
        opt_states = [opt.init(p) for opt, p in zip(opts, params)]
        return params, opt_states, opts

    def loss(self, params: AllParams, data: Array, beta: float, em: bool) -> tuple[float, Any]:
        prior_params, *rec_params = params
        
        prior = self.model.prior.update(prior_params)
        # prior = stopgrad(self.model.prior.update(prior_params))

        ### E-step
        prior_chain, factors_nat, posterior = self.get_posterior(prior, rec_params, data)
        if em: posterior = stopgrad(posterior)

        ### M-step
        kl_qf, log_Gamma, kl_qp = self.get_loss_terms(prior_chain, factors_nat, posterior)

        Z = self.batch_size * self.num_timesteps * self.num_factors
        loss = -(log_Gamma - kl_qf - beta * kl_qp) / Z

        aux = {
            'posterior': posterior,
            'factors_nat': factors_nat,
            'kl_qp': kl_qp / Z,
            'kl_qf': kl_qf / Z,
            'log_Gamma': log_Gamma / Z
        }

        return loss, aux
    
    def get_posterior(self, prior, rec_params, data):
        factors_nat = self.model.get_factors(rec_params, data) # JxBxTxK
        factors_tot = AllParam(factors_nat.sum(axis=0)) # BxTxK
        prior_chain = prior.to_chain(self.num_timesteps) # TxK
        posterior = vmap(lambda f: parallel_smoother(prior, f, self.model.latent_dim))(factors_tot.dist_param) # BxTxK
        
        return prior_chain, factors_nat, posterior
    
        #TODO: implement flexible_vmap function for factors

    def get_loss_terms(self, prior_chain, factors_nat, posterior):
        kl_qp = vmap(lambda qtk: qtk.kl(prior_chain))(posterior) # B

        prior_chain = prior_chain.all_param
        posterior = posterior.all_param

        kl_qf = vmap(lambda fntk: vmap(vmap(lambda qtk, ftk: qtk.kl(qtk+ftk)))(posterior.nat_param, fntk))(factors_nat) # JxBxT
        log_gammas = vmap(lambda fntk:
                          vmap(lambda qnk, fnk, pk:
                               vmap(lambda qk:
                                    vmap(lambda fk: (fk + qk).lognormalizer - (fk + pk).lognormalizer)(fnk)
                                )(qnk),
                               in_axes=(1,1,0)
                            )(posterior.nat_param, fntk, prior_chain.nat_param)
                         )(factors_nat) # JxTxBxB
        
        log_Gamma = vmap(vmap(
            lambda G: jnp.diag(G) - logsumexp(G, axis=1)
        ))(log_gammas) # JxTxB

        return kl_qf.sum(), log_Gamma.sum(), kl_qp.sum()


def parallel_smoother(prior, factors, latent_dim):
    C = jnp.eye(latent_dim)
    d = jnp.zeros(latent_dim) # TODO: check if C,d are automatically initialized to these values

    lgssm_params = make_lgssm_params(
        prior.params['m1'],
        prior.params['Q1'],
        prior.params['A'],
        prior.params['Q'],
        C,
        factors.params['cov'],
        dynamics_bias=prior.params['b'],
        emissions_bias=d
    )
    smoother_out = parallel_lgssm_smoother(lgssm_params, factors.params['mean'])._asdict()

    # in rare cases the smoothed covariances have min. evalues ~-1e-6, so add a correction to be safe (checked that this has no effect on experiments that were already stable)
    smoother_out['smoothed_covariances'] += 1e-5 * jnp.eye(latent_dim)

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
