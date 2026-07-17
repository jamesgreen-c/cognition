import jax
import jax.numpy as jnp
import jax.random as jr

from functools import partial

from jax import Array, vmap
from jax.random import PRNGKey

from rp_ssm.utils.dataset import Dataset


class LDSPrior:
    def __init__(self, T):
        self.T = T

    def sample(self, key, stationary_params, num_samples):
        m1 = stationary_params['m1']
        Q1 = stationary_params['Q1']
        A = stationary_params['A']
        b = stationary_params['b']
        Q = stationary_params['Q']

        As = jnp.tile(A[None], (self.T, 1, 1))
        bs = jnp.concatenate([
            m1[None],
            jnp.tile(b[None], (self.T-1, 1))
        ])
        Qs = jnp.concatenate([
            Q1[None],
            jnp.tile(Q[None], (self.T-1, 1, 1))
        ])

        @partial(jnp.vectorize, signature="(n),(t,d,d),(t,d),(t,d,d)->(t,d)")
        def sample_single(key, A, b, Q):
            biases = jr.multivariate_normal(key, b, Q)
            init_elems = (A, biases)

            @vmap
            def recursion(elem1, elem2):
                A1, b1 = elem1
                A2, b2 = elem2
                return A2 @ A1, A2 @ b1 + b2
            
            _, sample = jax.lax.associative_scan(recursion, init_elems)
            return sample

        keys = jr.split(key, num_samples)
        sample = sample_single(keys, As, bs, Qs)

        return sample
    

def random_rotation(key, n, theta=None):
    if n == 1:
        return jnp.eye(1) * jr.uniform(key)
    if theta is None:
        key, subkey = jr.split(key)
        theta = 0.5 * jnp.pi * jr.uniform(subkey)
    rot = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                     [jnp.sin(theta), jnp.cos(theta)]])
    out = jnp.eye(n)
    out = out.at[:2, :2].set(rot)
    q = jnp.linalg.qr(jr.uniform(key, shape=(n, n)))[0]
    return q.dot(out).dot(q.T)


def get_data(
    key: PRNGKey,
    num_factors: int,
    latent_dim: int,
    emission_dim: int,
    num_sequences: int,
    num_timesteps: int,
    emission_cov: float = 0.1,
) -> Dataset:
    
    print(f"""
Generating data:
    - Data modalities:  {num_factors}
    - Latent dim:       {latent_dim}
    - Emission dim:     {emission_dim}
    - Sequences:        {num_sequences}
    - Timesteps:        {num_timesteps}\n
""" )

    J, D, K, N, T = num_factors, latent_dim, emission_dim, num_sequences, num_timesteps

    key, key_m1, key_A, key_b, key_C, key_d = jr.split(key, 6)

    A = 0.95 * random_rotation(key_A, D, theta=jnp.pi/5) ### 0.95 ### theta
    Q = jnp.eye(D) - A @ A.T

    params = {
        'm1': jnp.zeros(D),
        'Q1': jnp.eye(D),
        'A': A,
        'b': jnp.zeros(D),
        'Q': Q,
        'R': jnp.eye(K) * jnp.sqrt(emission_cov) # keep R constant across J for now
    }

    lat_key, obs_key = jr.split(key)

    # sample an additional 25% sequences to use as validation data
    M = N + N // 4

    prior = LDSPrior(T)
    latent_sample = prior.sample(lat_key, params, M) # MxTxK

    emission_params = {
        'C': jr.normal(key_C, shape=(J, K, D)),
        'd': jr.normal(key_d, shape=(J, K)),
    }

    # not a pure function (uses M,T,latent_sample) but should be fine
    def sample_single_factor(emission_params, key):
        C, d, R = emission_params['C'], emission_params['d'], params['R']
        obs_sample = latent_sample @ C.T + jr.multivariate_normal(key, d, R, shape=(M, T)) # MxTxD
        return obs_sample

    obs_samples = vmap(sample_single_factor)(emission_params, jr.split(obs_key, J))

    params.update(emission_params)
    
    return Dataset(
        train_data=tuple([o[:N] for o in obs_samples]),
        train_states=latent_sample[:N],
        val_data=tuple([v[N:] for v in obs_samples]),
        val_states=latent_sample[N:],
        params=params
    )