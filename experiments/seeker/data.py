import jax.numpy as jnp
import jax.random as jr

from jax import Array
from jax.random import PRNGKey
from jax.scipy.stats import uniform

from rp_slac.environment import Environment


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
        # return jnp.array([
        #     position,
        #     velocity,
        #     position**2,
        #     jnp.sin(position),
        #     jnp.cos(position),
        # ])
        return state

    def transition(self, key: PRNGKey, state: Array, action: Array):
        position, velocity = state
        # action = jnp.clip(jnp.squeeze(action), -1.0, 1.0)
        action = jnp.squeeze(action)
        
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
        reward -= 0.1 * velocity**2
        reward -= 0.05 * action**2
        return next_state, reward

    def random_action(self, key: PRNGKey, state):
        """
        take random actions initially. Must be atleast 1D
        """
        action = jr.uniform(key, (1,), minval=-1.0, maxval=1.0)
        log_prob = uniform.logpdf(action, loc=-1, scale=2).sum()
        return action, log_prob

    def is_terminal_state(self, state):
        """Reset the environment when position leaves [-5, 5]."""
        position, _ = state
        return jnp.abs(position) >= 5.0

    def grid(self, num_positions: int = 101, num_velocities: int = 101):
        """Make an observation grid over positions and velocities."""
        positions = jnp.linspace(-5.0, 5.0, num_positions)
        velocities = jnp.linspace(-1.0, 1.0, num_velocities)

        position_grid, velocity_grid = jnp.meshgrid(positions, velocities, indexing="ij")

        # observation_grid = jnp.stack(
        #     [
        #         position_grid,
        #         velocity_grid,
        #         position_grid**2,
        #         jnp.sin(position_grid),
        #         jnp.cos(position_grid),
        #     ],
        #     axis=-1,
        # )
        observation_grid = jnp.stack([position_grid, velocity_grid], axis=-1)

        return observation_grid
    