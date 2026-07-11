import jax.numpy as jnp


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