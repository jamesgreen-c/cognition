import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Callable

from jax import Array, lax, vmap
from jax.random import PRNGKey

from rp_ssm.utils.dataset import Dataset
from rp_control.environment import Environment


# ===== helper function =====

def angle_normalise(theta):
    return (theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

def random_policy(key, state):
    """
    take random actions initially. Must be atleast 1D
    """
    return jr.uniform(key, (1,), minval=-1.0, maxval=1.0)


# ===== data generation =====
class CentreSeekingEnvironment(Environment):

    def __init__(
            self,
            T: int,
            velocity_decay: float = 0.9,
            action_scale: float = 0.2,
            velocity_scale: float = 0.1,
            process_std: float = 0.01,
    ):
        super().__init__()
        self.T = T
        self.velocity_decay = velocity_decay
        self.action_scale = action_scale
        self.velocity_scale = velocity_scale
        self.process_std = process_std

    def initial_state(self, key: PRNGKey):
        position_key, velocity_key = jr.split(key)
        position = jr.uniform(position_key, (), minval=-2.0, maxval=2.0)
        velocity = 0.1 * jr.normal(velocity_key)
        return jnp.array([position, velocity])

    def observe(self, key: PRNGKey, state: Array):
        position, velocity = state
        return jnp.array([
            position,
            velocity,
            position**2,
            jnp.sin(position),
            jnp.cos(position),
        ])

    def transition(self, key: PRNGKey, state: Array, action: Array):
        position, velocity = state
        action = jnp.clip(jnp.squeeze(action), -1.0, 1.0)

        velocity = self.velocity_decay * velocity
        velocity += self.action_scale * action
        velocity += self.process_std * jr.normal(key)

        position = position + self.velocity_scale * velocity
        return jnp.array([position, velocity])

    def model(self, key: PRNGKey, state: Array, action: Array):
        next_state = self.transition(key, state, action)
        position, velocity = next_state
        action = jnp.squeeze(action)

        reward = -position**2
        reward -= 0.05 * velocity**2
        reward -= 0.001 * action**2
        return next_state, reward
    

def get_data(
        key: PRNGKey,
        policy: Callable,
        num_factors: int,
        num_sequences: int,
        num_timesteps: int,
        emission_cov: float = 0.1,
    ) -> Dataset:

    print(f"""
Generating pendulum data:
    - Num factors:      {num_factors}
    - Sequences:        {num_sequences}
    - Timesteps:        {num_timesteps}
""")

    J, N, T = num_factors, num_sequences, num_timesteps
    M = N + N // 4      # 25% extra for validation data

    # initialise
    environment = CentreSeekingEnvironment(T=T)

    # sample latent states using policy
    _, latent_sample, obs_samples, actions = environment.sample(
        key,
        policy,
        num_samples=M,
    )
    
    return Dataset(
        train_data=(obs_samples[:N], ),
        train_states=latent_sample[:N],
        val_data=(obs_samples[N:], ),
        val_states=latent_sample[N:],
        params={
            "train_actions": actions[:N],
            "val_actions": actions[N:],
        },
    )
