import jax
import jax.numpy as jnp
import flax.linen as nn

from jax import vmap, Array
from jax.scipy.linalg import solve_triangular

from rpm.utils.math import spd_inverse, inv_quad_form_symmetric, inv_quad_form
from rpm.utils.vmapping import auto_vmap

from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN


class GaussianNatParam:
    """
    Gaussian natural parameters, given by ['p', 'pwm'] = [A*m, A],
    where A is the precision matrix and m is the mean. The sufficient
    statistics are [-0.5*xx.T, x].
    """
    def __init__(self, p: Array, pwm: Array):
        self.p = p
        self.pwm = pwm

    @property
    def latent_dim(self):
        return self.pwm.shape[-1]

    @property
    @auto_vmap('pwm', 1)
    def dist_param(self) -> "GaussianDistParam":
        cov = spd_inverse(self.p)
        mean = cov @ self.pwm
        return GaussianDistParam(
            mean=mean,
            cov=cov
        )

    @property
    def lognormalizer(self):
        quad, det = inv_quad_form_symmetric(self.p, self.pwm)
        return 0.5 * (quad - det)
    
    def expsuffstat_dot(self, v):
        term1 = inv_quad_form(self.p, self.pwm, v.pwm)
        L = jnp.linalg.cholesky(self.p)
        first = solve_triangular(
            L.T,
            solve_triangular(L, v.p, lower=True),
            lower=False
        )
        second = solve_triangular(
            L.T,
            solve_triangular(L, jnp.outer(self.pwm, self.pwm), lower=True),
            lower=False
        )
        term2 = -0.5 * jnp.trace(first + second @ first)
        return term1 + term2
    

class GaussianDistParam:
    """Contains parameters ['mean', 'cov']"""

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    @property
    @auto_vmap('mean', 1)
    def nat_param(self) -> GaussianNatParam:
        p = spd_inverse(self.cov)
        pwm = p @ self.mean
        return GaussianNatParam(
            p=p,
            pwm=pwm
        )
    
