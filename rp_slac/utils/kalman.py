"""Dynamax filtering/smoothing adapters for Gaussian recognition factors.

The pseudo-observation at t is the recognition factor's Gaussian mean, with
its covariance as observation noise and identity emission matrix. Both
entry points use the same LGSSM and return the same causal sampling object.
"""

import jax.numpy as jnp
from jax import vmap

from dynamax.linear_gaussian_ssm import parallel_lgssm_filter, parallel_lgssm_smoother
from dynamax.linear_gaussian_ssm.inference import make_lgssm_params
from dynamax.utils.utils import psd_solve, symmetrize

from rp_slac.distributions import LGChainDistParam, LGFilterDistParam
from rp_slac.utils.math import inv_quad_form


def _lgssm_params(prior, factors, actions):
    """Build one unbatched, action-conditioned Dynamax model."""

    times, dim = factors.params['mean'].shape
    action_dim = prior.params['B'].shape[-1]

    if actions is None:
        if action_dim:
            raise ValueError('Recorded actions are required for this model.')
        actions = jnp.zeros((times - 1, 0), dtype=factors.params['mean'].dtype)
    if actions.ndim == 1 and action_dim == 1:
        actions = actions[:, None]
    if actions.shape != (times - 1, action_dim):
        raise ValueError('Expected unbatched actions shaped (T-1,action_dim).')

    # Dynamax parameters have a time axis of length T. The final transition
    # is unused; padding also makes T=1 safe when its vmaps are traced.
    # biases = jnp.concatenate((biases, jnp.zeros((1, dim), dtype=biases.dtype)))
    
    biases = actions @ prior.params['B'].T + prior.params['b']
    return make_lgssm_params(
        prior.params['m1'], 
        prior.params['Q1'],
        prior.params['A'], 
        prior.params['Q'],
        jnp.eye(dim, dtype=biases.dtype), 
        factors.params['cov'],
        dynamics_bias=biases,
        emissions_bias=jnp.zeros(dim, dtype=biases.dtype),
    )


def _filter_distribution(prior, factors, output):
    """Choose a forward coupling while preserving Dynamax's filter marginals.

    With prediction covariance S_t and observation covariance R_t,
    K_t = S_t (S_t + R_t)^{-1}. Apply the usual Kalman correction to each
    prior draw, with an independent observation perturbation of covariance
    R_t. The resulting conditional has

        F_t = (I-K_t) A,
        V_t = (I-K_t) Q (I-K_t).T + K_t R_t K_t.T.

    This Joseph form is positive semidefinite and avoids subtracting marginal
    covariances. It defines the coupling; it is not a smoothing operation.
    """
    A = prior.params['A']
    Q = prior.params['Q']

    means = output.filtered_means
    covs = symmetrize(output.filtered_covariances)
    predicted_covs = A @ covs[:-1] @ A.T + Q
    observation_covs = factors.params['cov'][1:]
    gains = vmap(lambda s, r: psd_solve(s + r, s.T).T)(predicted_covs, observation_covs)

    residual = jnp.eye(A.shape[-1], dtype=A.dtype) - gains
    weights = residual @ A
    noise = residual @ Q @ jnp.swapaxes(residual, -1, -2)
    noise += gains @ observation_covs @ jnp.swapaxes(gains, -1, -2)

    return LGFilterDistParam(
        mean=means, 
        cov=covs,
        conditional_weights=weights, 
        conditional_covs=symmetrize(noise),
    )


def parallel_filter(prior, factors, actions):
    """Return an unbatched LGFilterDistParam using only a Dynamax forward pass.

    factors is a GaussianDistParam with mean (T,D) and cov (T,D,D).
    Actions have shape (T-1,A); None is accepted for action-free models.
    """
    params = _lgssm_params(prior, factors, actions)
    output = parallel_lgssm_filter(params, factors.params['mean'])
    return _filter_distribution(prior, factors, output)


def parallel_smoother(prior, factors, actions):
    """Return (smoothing LGChainDistParam, causal LGFilterDistParam).

    Reuse the forward output already computed by Dynamax's smoother, rather
    than running a second filter. Only the smoothing chain enters chain KLs.
    """
    params = _lgssm_params(prior, factors, actions)
    output = parallel_lgssm_smoother(params, factors.params['mean'])
    filtered = _filter_distribution(prior, factors, output)

    A = prior.params['A']
    Q = prior.params['Q']

    filtered_covs = symmetrize(output.filtered_covariances)
    smoothed_covs = symmetrize(output.smoothed_covariances)
    cross_covs = vmap(
        lambda S, F: inv_quad_form(Q + A @ F @ A.T, S.T, A @ F)
    )(smoothed_covs[1:], filtered_covs[:-1])

    posterior = LGChainDistParam(
        means=output.smoothed_means, 
        covs=smoothed_covs, 
        cross_covs=cross_covs,
    )
    return posterior, filtered
