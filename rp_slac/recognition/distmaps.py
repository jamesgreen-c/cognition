import jax.numpy as jnp
import flax.linen as nn

from jax import Array
from rp_slac.distributions import NatParam, GaussianNatParam


EPS = 1e-3


class DistMap:
    def __init__(self, latent_dim: int):
        self.latent_dim = latent_dim

    @property
    def input_dim(self):
        """
        Returns the dimension of the input to the distmap.
        E.g. if latent_dim = 3 and then distmap gives
        a Gaussian with diagonal precision, then the input
        dim should be 3+3=6. If the precision were Cholesky-
        parametrized, then the input dim should be 3+3*(3+1)/2=9.
        """
        pass

    def __call__(self, x: Array) -> NatParam:
        pass


class MVNDiag(DistMap):

    @property
    def input_dim(self):
        return self.latent_dim * 2
    
    def __call__(self, x: Array) -> GaussianNatParam:
        pwm = x[:self.latent_dim]
        p = jnp.diag(nn.softplus(x[self.latent_dim:]))
        
        return GaussianNatParam(p=p, pwm=pwm)


class SLACMVNDiag(DistMap):
    """Diagonal Gaussian parametrised by mean and clipped log-standard-deviation."""

    def __init__(
            self,
            latent_dim: int,
            min_log_std: float = -20.0,
            max_log_std: float = 2.0,
        ):
        super().__init__(latent_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    @property
    def input_dim(self):
        return self.latent_dim * 2

    def __call__(self, x: Array) -> GaussianNatParam:
        mean = x[:self.latent_dim]
        log_std = x[self.latent_dim:]

        # bound the standard deviation
        log_std = jnp.clip(log_std, self.min_log_std, self.max_log_std)

        # convert to natural parameters
        precision_diag = jnp.exp(-2.0 * log_std)
        p = jnp.diag(precision_diag)
        pwm = precision_diag * mean

        return GaussianNatParam(p=p, pwm=pwm)


class MVNCholesky(DistMap):

    @property
    def input_dim(self):
        return self.latent_dim + self.latent_dim * (self.latent_dim + 1) // 2
    
    def __call__(self, x: Array) -> GaussianNatParam:
        pwm = x[:self.latent_dim]
        p_lower_tri = x[self.latent_dim:2*-self.latent_dim]
        p_diag = x[-self.latent_dim:]
    
        L = jnp.zeros((self.latent_dim, self.latent_dim))
        L = L.at[jnp.diag_indices(self.latent_dim)].set(p_diag)
        L = L.at[jnp.tril_indices(self.latent_dim, -1)].set(p_lower_tri)
        p = L @ L.T + EPS * jnp.eye(self.latent_dim)

        return GaussianNatParam(p=p, pwm=pwm)


class MVNCholeskySoftplus(DistMap):

    @property
    def input_dim(self):
        return self.latent_dim + self.latent_dim * (self.latent_dim + 1) // 2
    
    def __call__(self, x: Array) -> GaussianNatParam:
        pwm = x[:self.latent_dim]
        p_lower_tri = x[self.latent_dim:-self.latent_dim]
        p_diag = x[-self.latent_dim:]
    
        L = jnp.zeros((self.latent_dim, self.latent_dim))
        L = L.at[jnp.diag_indices(self.latent_dim)].set(nn.softplus(p_diag))
        L = L.at[jnp.tril_indices(self.latent_dim, -1)].set(p_lower_tri)
        p = L @ L.T + EPS * jnp.eye(self.latent_dim)

        return GaussianNatParam(p=p, pwm=pwm)


class MVNCholeskyReLU(DistMap):

    @property
    def input_dim(self):
        return self.latent_dim + self.latent_dim * (self.latent_dim + 1) // 2
    
    def __call__(self, x: Array) -> GaussianNatParam:
        pwm = x[:self.latent_dim]
        p_lower_tri = x[self.latent_dim:-self.latent_dim]
        p_diag = x[-self.latent_dim:]
    
        L = jnp.zeros((self.latent_dim, self.latent_dim))
        L = L.at[jnp.diag_indices(self.latent_dim)].set(nn.relu(p_diag))
        L = L.at[jnp.tril_indices(self.latent_dim, -1)].set(p_lower_tri)
        p = L @ L.T + EPS * jnp.eye(self.latent_dim)

        return GaussianNatParam(p=p, pwm=pwm)
