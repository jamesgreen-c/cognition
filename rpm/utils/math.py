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
