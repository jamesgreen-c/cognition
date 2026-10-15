import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Any, Union, Tuple, Optional

from jax import vmap
from jax.random import PRNGKey, multivariate_normal
from jax.tree_util import tree_map
from jax.scipy.linalg import solve_triangular
from flax.core.frozen_dict import FrozenDict

from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

from rp_slac.utils.vmapping import auto_vmap
from rp_slac.utils.math import spd_inverse, inv_quad_form, inv_quad_form_symmetric, gaussian_kl_precision



from typing import TYPE_CHECKING, Sequence
if TYPE_CHECKING:
    from rp_slac.recognition.distmaps import DistMap


NetworkParams = Union[FrozenDict, dict[str, Any]]
AllParams = Tuple['LGStationaryParam', *Tuple[NetworkParams, ...]]


class NatParam:
    """
    Class for storing and manipulating natural
    parameters of an exponential family distribution.

    Initialize the class by passing in parameters
    with names, e.g. for a Gaussian distribution,
    dist = NatParam(precision=..., precision_weighted_mean=...).

    The class then saves the parameters as a dict, e.g.
    in the Gaussian case above,
    dist.params = {'precision': ..., 'precision_weighted_mean': ...}.

    When defining a custom NatParam class, the functions that need to
    be specified are lognormalizer, expsuffstat_dot, and flatten
    (and optionally dist_param).

    Every subclass of NatParam is registered as a PyTree node,
    which allows the use of vmap and jit on functions with
    signature Array -> (subclass of NatParam).
    """
    
    def __init__(self, dist_map: Optional["DistMap"] = None, **kwargs):
        self.dist_map = dist_map
        # TODO: implement this
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
        self.params = kwargs

    def __add__(self, other):
        return type(self)(self.dist_map, **{k: v + other.params[k] for k, v in self.params.items()})

    def __sub__(self, other):
        return type(self)(self.dist_map, **{k: v - other.params[k] for k, v in self.params.items()})
    
    def sum(self, axis):
        return type(self)(self.dist_map, **{k: jnp.sum(v, axis=axis) for k, v in self.params.items()})

    @property
    def lognormalizer(self):
        pass

    def expsuffstat_dot(self, v):
        """Compute the dot product <t(x)>*v, where v is a NatParam of the same type."""
        pass

    def kl(self, other):
        return (self.expsuffstat_dot(self - other) - self.lognormalizer + other.lognormalizer)
    
    def update(self, params):
        flattened_params = self.flatten(params)
        return self.dist_map(flattened_params)

    @property
    def latent_dim(self):
        pass

    @property
    def dist_param(self):
        """Turn natural parameters into corresponding distribution parameters"""
        pass

    ##### register subclasses of NatParam as PyTree nodes for JAX operations
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node_class(cls)

    def tree_flatten(self):
        leaves = list(self.params.values())
        aux_data = (list(self.params.keys()), self.dist_map)
        return leaves, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        param_keys, dist_map = aux_data
        return cls(dist_map, **dict(zip(param_keys, leaves)))
    #####
    

class DistParam:
    """
    Generic class for standard distribution parameters
    (e.g. mean and covariance for a Gaussian). Contains
    a method self.nat_param that returns the corresponding
    NatParam object.
    """
    def __init__(self, dist_map: Optional["DistMap"] = None, **kwargs):
        self.dist_map = dist_map
        self.params = kwargs

    # TODO: can remove--just for debugging
    def __add__(self, other):
        return type(self)(self.dist_map, **{k: v + other.params[k] for k, v in self.params.items()})

    @property
    def nat_param(self):
        pass

    ##### register subclasses of NatParam as PyTree nodes for JAX operations
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node_class(cls)

    def tree_flatten(self):
        leaves = list(self.params.values())
        aux_data = (list(self.params.keys()), self.dist_map)
        return leaves, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        param_keys, dist_map = aux_data
        return cls(dist_map, **dict(zip(param_keys, leaves)))
    #####


class AllParam:
    """
    Class storing both natural and distribution parameters
    of a distribution. In the RP-SSM code, the prior, factors,
    and posterior all require both natural and distribution
    parameters to compute the loss. This class is initialized
    with either the natural or distribution parameters, and
    automatically computes the other on initialization.
    """
    nat_param: NatParam
    dist_param: DistParam

    def __init__(self, param):
        if isinstance(param, NatParam):
            self.nat_param = param
            self.dist_param = param.dist_param
        elif isinstance(param, DistParam):
            self.nat_param = param.nat_param
            self.dist_param = param
        elif isinstance(param, tuple):
            self.nat_param = param[0]
            self.dist_param = param[1]

    def kl(self, other):
        return self.nat_param.kl(other.nat_param)

    @property
    def lognormalizer(self):
        return self.nat_param.lognormalizer

    def __add__(self, other):
        return AllParam((self.nat_param + other.nat_param, self.dist_param + other.dist_param))


class GaussianNatParam(NatParam):
    """
    Gaussian natural parameters, given by ['p', 'pwm'] = [A*m, A],
    where A is the precision matrix and m is the mean. The sufficient
    statistics are [-0.5*xx.T, x].
    """
    # p: Array
    # pwm: Array

    # def kl(self, other):
    #     """Gaussian KL using mean differences and relative Cholesky factors."""
    #     return gaussian_kl_precision(
    #         self.params['p'], self.params['pwm'], other.params['p'], other.params['pwm']
    #     )

    def kl(self, other):
        """More numerically stable KL divergence function for Gaussians"""
        q = self.dist_param
        p = other.dist_param

        delta = q.params["mean"] - p.params["mean"]
        precision_p = other.params["p"]

        trace = jnp.trace(precision_p @ q.params["cov"], axis1=-2, axis2=-1)
        quadratic = jnp.einsum("...i,...ij,...j->...", delta, precision_p, delta)

        logdet_q = jnp.linalg.slogdet(self.params["p"])[1]
        logdet_p = jnp.linalg.slogdet(precision_p)[1]

        return 0.5 * (trace + quadratic - self.latent_dim + logdet_q - logdet_p)

    def flatten(self, params):
        assert {'pwm', 'chol'} <= set(params)
        return jnp.concatenate([params['pwm'], params['chol']])

    @property
    def latent_dim(self):
        return self.params['pwm'].shape[-1]

    @property
    @auto_vmap('pwm', 1)
    def dist_param(self):
        cov = spd_inverse(self.params['p'])
        mean = cov @ self.params['pwm']
        return GaussianDistParam(
            dist_map=None,
            mean=mean,
            cov=cov
        ) # set dist_map=None because it won't be necessary anymore

    @property
    def lognormalizer(self):
        quad, det = inv_quad_form_symmetric(self.params['p'], self.params['pwm'])
        return 0.5 * (quad - det)
    
    def expsuffstat_dot(self, v):
        term1 = inv_quad_form(self.params['p'], self.params['pwm'], v.params['pwm'])
        L = jnp.linalg.cholesky(self.params['p'])
        first = solve_triangular(
            L.T,
            solve_triangular(L, v.params['p'], lower=True),
            lower=False
        )
        second = solve_triangular(
            L.T,
            solve_triangular(L, jnp.outer(self.params['pwm'], self.params['pwm']), lower=True),
            lower=False
        )
        term2 = -0.5 * jnp.trace(first + second @ first)
        return term1 + term2
    

class GaussianDistParam(DistParam):
    """Contains parameters ['mean', 'cov']"""
    # mean: Array
    # cov: Array

    @property
    @auto_vmap('mean', 1)
    def nat_param(self) -> GaussianNatParam:
        p = spd_inverse(self.params['cov'])
        pwm = p @ self.params['mean']
        return GaussianNatParam(
            dist_map=None,
            p=p,
            pwm=pwm
        )
    
    def sample(self, key: jax.Array, shape: Sequence[int]):
        return jax.random.multivariate_normal(
            key, self.params["mean"], self.params["cov"], shape
        )

    def log_prob(self, x):
        """
        Compute the log-probability of x under the
        Gaussian distribution `self`.
        """
        assert x.shape == self.params["mean"].shape
        if self.params["cov"].ndim == 2:
            inv_quad_form, logdet = inv_quad_form_symmetric(
                self.params["cov"], x - self.params["mean"]
            )
        elif self.params["cov"].ndim == 1:
            logdet = jnp.sum(jnp.log(self.params["cov"]))
            inv_quad_form = jnp.sum((x - self.params["mean"]) ** 2 / self.params["cov"])
        return -0.5 * (
            self.params["mean"].shape[-1] * jnp.log(2.0 * jnp.pi) + logdet + inv_quad_form
        )
    

@jax.tree_util.register_pytree_node_class
class LGStationaryParam:
    """
    Class containing parameters defining a stationary
    linear-Gaussian chain. Contains m1,Q1,A,b,Q, where
    p(z_1) = N(m1,Q1) and p(z_t+1|z_t) = N(Az_t+b,Q).

    TODO: currently fixed p(z_1)=N(0,I) and sets
    Q=I-AA.T to enforce stationarity and stability.

    TODO: add init function
    """
    # A: Array
    # b: Array

    # TODO: if p(z_1) is invariant dist, don't do the recursion, just copy p(z_1) T times

    def __init__(self, stationary: bool, **kwargs):
        self.params = kwargs
        self.stationary = stationary
        dim = kwargs['A'].shape[0]

        # when initialized at stationary distribution, WLOG assume m1=0, Q1=I
        if stationary:
            self.params.update(
                {
                    'm1': jnp.zeros(dim),
                    'Q1': jnp.eye(dim)
                }
            )
            self.opt_param = {'A': self.params['A'], 'B': self.params['B']}

        else:
            self.opt_param = {
                'm1': self.params['m1'],
                'Q1': self.params['Q1'],
                'A': self.params['A'],
                'B': self.params['B']
            }
            
        # if A is given as a vector, reshape it to a diagonal matrix
        if kwargs['A'].ndim == 1:
            self.params.update({'A': jnp.diag(kwargs['A'])})

        # enforce stability
        self.params.update(
            {
                'Q': self.params['Q1'] - self.params['A'] @ self.params['Q1'] @ self.params['A'].T,
                'b': (jnp.eye(dim) - self.params['A']) @ self.params['m1']
            }
        )

    @property
    def latent_dim(self):
        return self.params['A'].shape[0]
    
    def update(self, params):
        """Update an arbitrary number of parameters."""
        new_params = {**self.params, **params} # jax-compatible update
        # TODO: when not stationary, turn Q1 param into covariance
        if new_params['A'].ndim == 1:
            new_params['A'] = nn.sigmoid(params['A'])
        return LGStationaryParam(stationary=self.stationary, **new_params)
    
    def to_chain(self, actions, m1=None):
        """
        Construct p(z_1:T | a_1:T-1) as an LGChainDistParam.

        Parameters
        ----------
        actions:
            Unbatched action sequence with shape (T - 1, action_dim).

        Returns
        -------
        LGChainDistParam with:
            means:       (T, latent_dim)
            covs:        (T, latent_dim, latent_dim)
            cross_covs:  (T - 1, latent_dim, latent_dim)
        """
        if actions.ndim == 1:
            actions = actions[:, None]

        num_timesteps = actions.shape[0] + 1

        m1 = self.params["m1"] if m1 is None else m1
        Q1 = self.params["Q1"]
        A = self.params["A"]
        B = self.params["B"]
        b = self.params["b"]
        Q = self.params["Q"]

        As = jnp.tile(A[None], (num_timesteps, 1, 1))
        Qs = jnp.concatenate([Q1[None], jnp.tile(Q[None], (num_timesteps-1, 1, 1))])

        # c_t = B a_t + b for p(z_{t+1} | z_t, a_t)
        Cs = actions @ B.T + b
        bs = jnp.concatenate([m1[None], Cs])

        @vmap
        def recursion(elem1, elem2):
            A1, b1, Q1 = elem1
            A2, b2, Q2 = elem2
            return A2 @ A1, A2 @ b1 + b2, A2 @ Q1 @ A2.T + Q2

        init_elems = (As, bs, Qs)
        _, means, covs = jax.lax.associative_scan(recursion, init_elems)
        cross_covs = As[1:] @ covs[:-1] # Cov(z_t+1,z_t) = A_t+1 @ Sigma_t
        
        return LGChainDistParam(means=means, covs=covs, cross_covs=cross_covs)

    def tree_flatten(self):
        leaves = list(self.params.values())
        aux_data = (list(self.params.keys()), self.stationary)
        return leaves, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        param_keys, stationary = aux_data
        return cls(stationary=stationary, **dict(zip(param_keys, leaves)))
    

class LGChainDistParam(DistParam):
    """
    Class containing the full distribution of a linear-Gaussian chain.

    Contains parameters ['means', 'covs', 'cross_covs'].

    Is generated from LGStationaryParam.to_chain.

    Contains methods to compute KL divergences between chains and
    compute a corresponding AllParam object.
    """
    # means: Array
    # covs: Array
    # cross_covs: Array

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
            self.params['means'][1:-1],
            other.params['means'][1:-1],
            self.params['covs'][1:-1],
            other.params['covs'][1:-1]
        )
        pairwise = vmap(kl_tt)(
            self.params['means'][:-1], self.params['means'][1:],
            other.params['means'][:-1], other.params['means'][1:],
            self.params['covs'][:-1], self.params['covs'][1:],
            other.params['covs'][:-1], other.params['covs'][1:],
            self.params['cross_covs'], other.params['cross_covs']
        )

        return jnp.sum(pairwise) - jnp.sum(marginal)
    
    @property
    def all_param(self):
        dist_param = GaussianDistParam(
            dist_map=None,
            mean=self.params['means'],
            cov=self.params['covs']
        )
        return AllParam(dist_param)
    
    @property
    def mean_field(self):
        """Strip of all cross-covariances"""
        return GaussianDistParam(mean=self.params["means"], cov=self.params["covs"])

    # def sample(self, key: PRNGKey):
    #     """ Sample series from means and covs """
    #     return multivariate_normal(key, mean=self.params["means"], cov=self.params["covs"])

    def sample(self, key: PRNGKey):
        """Sample a joint trajectory using backward conditional sampling."""
        means = self.params["means"]
        covs = self.params["covs"]
        cross_covs = self.params["cross_covs"]

        def sample_trajectory(key, means, covs, cross_covs):
            T, latent_dim = means.shape
            keys = jax.random.split(key, T)
            I = jnp.eye(latent_dim)

            # sample z_{T-1} from its smoothed marginal
            z_T = multivariate_normal(keys[-1], mean=means[-1], cov=covs[-1])

            def backward_step(z_next, inputs):
                key, mean, next_mean, cov, next_cov, cross_cov = inputs

                # cross_cov = Cov(z_{t+1}, z_t)
                gain = jnp.linalg.solve(next_cov + 1e-6 * I, cross_cov).T

                conditional_mean = mean + gain @ (z_next - next_mean)
                conditional_cov = cov - gain @ cross_cov
                conditional_cov = 0.5 * (conditional_cov + conditional_cov.T)
                conditional_cov += 1e-6 * I

                z = multivariate_normal(key, mean=conditional_mean, cov=conditional_cov)
                return z, z

            inputs = (
                keys[:-1][::-1],
                means[:-1][::-1],
                means[1:][::-1],
                covs[:-1][::-1],
                covs[1:][::-1],
                cross_covs[::-1],
            )

            _, reverse_samples = jax.lax.scan(backward_step, z_T, inputs)
            return jnp.concatenate([reverse_samples[::-1], z_T[None]], axis=0)

        # unbatched: (T, D)
        if means.ndim == 2:
            return sample_trajectory(key, means, covs, cross_covs)

        # batched: (B, T, D)
        B = means.shape[0]
        keys = jax.random.split(key, B)
        return vmap(sample_trajectory)(keys, means, covs, cross_covs)


class LGFilterDistParam(DistParam):
    """A causal Gaussian path with Kalman-filter marginals.

    Parameters are 'mean', 'cov', 'conditional_weights', 'conditional_covs'.
    The first two hold m_t, P_t from Dynamax filtering. The last two specify
    the forward coupling

        z_0 ~ N(m_0, P_0),
        z_t | z_{t-1} ~ N(m_t + F_t (z_{t-1} - m_{t-1}), V_t).

    utils.kalman constructs F_t, V_t using a stochastic Kalman update, so
    F_t P_{t-1} F_t.T + V_t = P_t. This selects a Gaussian joint with the
    desired filtering marginals; those marginals alone do not define a joint.
    It is a causal surrogate path, not the full-data smoothing posterior or
    the product of locally normalised p(z_t | z_{t-1}) f_t(z_t) kernels.

    Shapes are (..., T, D), (..., T, D, D), and (..., T-1, D, D) for both
    conditional arrays. No chain KL calculation is provided.
    """

    # mean: Array
    # cov: Array
    # conditional_weights: Array
    # conditional_covs: Array

    @property
    def mean_field(self):
        return GaussianDistParam(mean=self.params['mean'], cov=self.params['cov'])

    @property
    def cross_covs(self):
        """Cov(z_{t+1}, z_t) under this particular forward coupling."""
        return self.params['conditional_weights'] @ self.params['cov'][..., :-1, :, :]

    def sample(self, key: jax.Array, *args, **kwargs):
        """Sample one joint trajectory per batch member, proceeding forward."""

        means = self.params['mean']
        covs = self.params['cov']
        weights = self.params['conditional_weights']
        noise_covs = self.params['conditional_covs']

        def sample_trajectory(key, means, covs, weights, noise_covs):

            keys = vmap(lambda t: jax.random.fold_in(key, t))(jnp.arange(means.shape[0]))
            initial = multivariate_normal(keys[0], means[0], covs[0])

            def step(previous, inputs):
                subkey, mean, previous_mean, weight, noise_cov = inputs
                conditional_mean = mean + weight @ (previous - previous_mean)
                state = multivariate_normal(subkey, conditional_mean, noise_cov)
                return state, state

            inputs = (keys[1:], means[1:], means[:-1], weights, noise_covs)
            _, states = jax.lax.scan(step, initial, inputs)
            return jnp.concatenate((initial[None], states), axis=0)

        if means.ndim == 2:
            return sample_trajectory(key, means, covs, weights, noise_covs)

        times, dim = means.shape[-2:]
        flat_means = means.reshape((-1, times, dim))
        flat_covs = covs.reshape((-1, times, dim, dim))

        # Specify the flattened batch explicitly, since T-1 can be zero.
        batch = flat_means.shape[0]
        flat_weights = weights.reshape((batch, times - 1, dim, dim))
        flat_noise = noise_covs.reshape((batch, times - 1, dim, dim))

        keys = vmap(lambda b: jax.random.fold_in(key, b))(jnp.arange(batch))
        samples = vmap(sample_trajectory)(keys, flat_means, flat_covs, flat_weights, flat_noise)
        return samples.reshape(means.shape)
