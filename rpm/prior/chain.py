import jax.numpy as jnp
from jax import vmap

from rpm.prior.gaussian import GaussianNatParam, GaussianDistParam
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN


class LinearGaussianChain:

    stationary: bool = True

    def __init__(self, means, covs, cross_covs):
        self.means = means
        self.covs = covs
        self.cross_covs = cross_covs

    def kl(self, other):
        """Compute KL(self||other)"""
        def kl_t(muq, mup, Sq, Sp):
            return MVN(muq, Sq).kl_divergence(MVN(mup, Sp))
        
        def kl_tt(
                muqt, muqtt,
                mupt, muptt,
                Sqt, Sqtt,
                Spt, Sptt,
                Sqx, Spx
            ):

            muq = jnp.concatenate((muqt, muqtt))
            mup = jnp.concatenate((mupt, muptt))
            Sq = jnp.block(
                [[Sqt, Sqx.T],
                 [Sqx,  Sqtt]]
            )
            Sp = jnp.block(
                [[Spt, Spx.T],
                 [Spx, Sptt]]
            )
            return kl_t(muq, mup, Sq, Sp)
        
        marginal = vmap(kl_t)(
            self.means[1:-1],
            other.means[1:-1],
            self.covs[1:-1],
            other.covs[1:-1]
        )
        pairwise = vmap(kl_tt)(
            self.means[:-1], self.means[1:],
            other.means[:-1], other.means[1:],
            self.covs[:-1], self.covs[1:],
            other.covs[:-1], other.covs[1:],
            self.cross_covs, other.cross_covs
        )

        return jnp.sum(pairwise) - jnp.sum(marginal)
    
    @property
    def dist_param(self) -> GaussianDistParam:
        return GaussianDistParam(
            mean=self.means,
            cov=self.covs
        )
    

def construct_chain(prior_params, T, stationary: bool = True) -> LinearGaussianChain:
    if stationary:
        means = jnp.tile(prior_params['m1'][None], (T, 1))
        covs = jnp.tile(prior_params['Q1'][None], (T, 1, 1))
        cross_covs = prior_params['A'] @ covs[:-1] # Cov(z_t+1,z_t) = A @ Sigma_t
        return LinearGaussianChain(means=means, covs=covs, cross_covs=cross_covs)
    
    else:
        raise NotImplementedError("construct_chain not implemented for non stationary chains")
