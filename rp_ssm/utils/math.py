import jax.numpy as jnp

from jax.scipy.linalg import solve_triangular


def inv_quad_form_symmetric(S, m):
    """Compute m.T * inv(S) * m and log(det(S))"""
    L = jnp.linalg.cholesky(S)
    x = solve_triangular(L, m, lower=True)
    logdet = 2 * jnp.sum(jnp.log(jnp.diag(L)))
    return x.T @ x, logdet


def inv_quad_form(S, m1, m2):
    """Compute m1.T * inv(S) * m2"""
    L = jnp.linalg.cholesky(S)
    x1 = solve_triangular(L, m1, lower=True)
    x2 = solve_triangular(L, m2, lower=True)
    return x1.T @ x2


def spd_inverse(S):
    """Invert a spd matrix via Cholesky"""
    L = jnp.linalg.cholesky(S)
    Linv = solve_triangular(L, jnp.eye(S.shape[-1]), lower=True)
    return Linv.T @ Linv


def logdet(S):
    L = jnp.linalg.cholesky(S)
    logdet = 2 * jnp.sum(jnp.log(jnp.diag(L)))
    return logdet


def scale_sv(A, eps):
    """
    Given a matrix A, scale its singular values to <=1-EPS.
    Only do so if there is a singular value already >1-EPS.

    Intuitively, this should work better than clipping
    if all singular values are relatively close to 1, as it
    preserves the relative sizes between singular values.
    If there is a single singular value that is very large,
    this will push the others to 0, so clipping may be
    more appropriate.
    """
    
    if len(A.shape) < 2:
        A = jnp.diag(A)  # from vector to diagonal matrix
        _, s, _ = jnp.linalg.svd(A)
        scale = jnp.maximum(jnp.max(s), 1-eps) / (1-eps)
        A = jnp.diag(A / scale)  # from diagonal matrix to vector
    else:
        _, s, _ = jnp.linalg.svd(A)
        scale = jnp.maximum(jnp.max(s), 1-eps) / (1-eps)
        A = A / scale
    return A


def clip_sv(A, eps):
    """
    Clip the SVs of a matrix to be less than 1-EPS.
    """
    u, s, vt = jnp.linalg.svd(A)
    return u @ jnp.diag(jnp.clip(s, 0., 1.-eps)) @ vt
