import jax
import jax.numpy as jnp
import flax.linen as nn

from jax import vmap, Array
from jax.lax import stop_gradient
from jax.scipy.special import logsumexp

from rpm.utils.math import inv_quad_form
from rpm.prior.chain import construct_chain, LinearGaussianChain

from dynamax.linear_gaussian_ssm import parallel_lgssm_smoother
from dynamax.linear_gaussian_ssm.inference import make_lgssm_params


@jax.jit(static_argnums=(0, 1))
def get_posterior(
        recognition,
        networks: tuple[nn.Module],
        params,
        data: tuple,
    ):
    """
    Runs the SMC sampler kernel to get posterior samples.

    Parameters
    ----------
    key:       RNG.
    networks:  Tuple of recognition neural networks to use.
    params:    Parameter dictionary with fields:
                - rec:    Recognition network parameters.
                - prior:  Prior model parameters.
    data:      PyTree of observations with leaves of shape (B, T, *_), where B is the
                    number of independent time-series and T is the number of time-steps.

    Returns
    -------
    factors:   Tuple/PyTree of recognition factors matching the latent-state factorisation.
                   Each factor has form (means, chols).
    posterior: The set of means, covs and cross_covs from Kalman smoothing on each observation.
    """
    # extract params
    rec_params = params["rec"]
    prior_params = params["prior"]

    # get posterior
    factors = tuple(recognition.apply(net, p, data) for net, p in zip(networks, rec_params))
    posterior = parallel_smoother(prior_params, factors, data)
    return factors, posterior


@jax.jit(static_argnums=(0, 1, 5))
def loss(
        recognition,
        networks: tuple[nn.Module],
        params: dict,
        data: Array,
        beta: float,
        em: bool
    ):

    data_leaf = jax.tree_util.tree_leaves(data)[0]
    B, T = data_leaf.shape[:2]

    # E-step
    factors, posterior = get_posterior(recognition, networks, params, data)
    if em: posterior = stop_gradient(posterior)

    # M-step
    kl_qf, log_Gammas, kl_qp = get_loss_terms(params["prior"], factors, posterior)
    loss = - (log_Gammas - kl_qf - beta * kl_qp) / (B * T)

    # pack return
    aux = {"factors": factors, "posterior": posterior}
    return loss, aux


def parallel_smoother(prior_params, factors, latent_dim):
    C = jnp.eye(latent_dim)
    d = jnp.zeros(latent_dim)

    # run Kalman smoothing on data batch
    lgssm_params = make_lgssm_params(
        prior_params['m1'],
        prior_params['Q1'],
        prior_params['A'],
        prior_params['Q'],
        C,
        factors.params['cov'],
        dynamics_bias=prior_params['b'],
        emissions_bias=d
    )
    smoother_out = parallel_lgssm_smoother(lgssm_params, factors.params['mean'])._asdict()
    smoother_out['smoothed_covariances'] += 1e-5 * jnp.eye(latent_dim)   # stability 

    # extract inference
    filtered_cov = smoother_out['filtered_covariances']
    smoothed_cov = smoother_out['smoothed_covariances']
    A, Q = prior_params['A'], prior_params['Q']

    # calculate cross covariances
    _cross_cov = lambda S, F: inv_quad_form(Q + A @ F @ A.T, S.T, A @ F)
    cross_covs = vmap(_cross_cov)(smoothed_cov[1:], filtered_cov[:-1])

    # return posterior
    posterior = LinearGaussianChain(
        means=smoother_out['smoothed_means'], 
        covs=smoothed_cov, 
        cross_covs=cross_covs
    )
    return posterior


def get_loss_terms(prior_params, factors, posterior: LinearGaussianChain):
    
    prior: LinearGaussianChain = construct_chain(prior_params)
    prior_natparams = prior.dist_param.nat_param
    posterior_natparams = posterior.dist_param.nat_param
    
    kl_qp = vmap(lambda qtk: qtk.kl(prior))(posterior) # B
    kl_qf = vmap(lambda fntk: vmap(vmap(lambda qtk, ftk: qtk.kl(qtk+ftk)))(posterior_natparams, fntk))(factors) # JxBxT
    log_gammas = vmap(lambda fntk:
                        vmap(lambda qnk, fnk, pk:
                            vmap(lambda qk:
                                vmap(lambda fk: (fk + qk).lognormalizer - (fk + pk).lognormalizer)(fnk)
                            )(qnk),
                            in_axes=(1,1,0)
                        )(posterior_natparams, fntk, prior_natparams)
                        )(factors) # JxTxBxB
    
    log_Gamma = vmap(vmap(
        lambda G: jnp.diag(G) - logsumexp(G, axis=1)
    ))(log_gammas) # JxTxB

    return kl_qf.sum(), log_Gamma.sum(), kl_qp.sum()