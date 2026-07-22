import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Callable

from jax import Array, lax, vmap
from jax.random import PRNGKey

from rp_ssm.utils.dataset import Dataset
from slac.environment import Environment


# ===== helper function =====

def angle_normalise(theta):
    return (theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


# ===== data generation =====

class PendulumEnvironment(Environment):

    physics_steps: int = 10

    def __init__(
        self,
        T: int,
        dt: float = 0.01,
        cart_mass: float = 1.0,
        pole_mass: float = 0.2,
        pole_length: float = 0.6,
        force_scale: float = 2.0,
        gravity: float = 9.81,
        cart_friction: float = 0.05,
        hinge_damping: float = 0.02,
        process_std: Array | None = None,
    ):  
        super().__init__()

        self.T = T
        self.dt = dt

        self.cart_mass = cart_mass
        self.cart_friction = cart_friction
        
        self.pole_mass = pole_mass
        self.pole_length = pole_length
        self.hinge_damping = hinge_damping

        self.force_scale = force_scale
        self.gravity = gravity

        # noise on cart and angular velocities if not provided.
        self.process_std = jnp.array([0.0, 0.002, 0.0, 0.005]) if process_std is None else process_std


    def model(self, key: PRNGKey, state: Array, action: Array):
        """
        Implement p(S', r | S, a)
        
        Parameters
        ---------- 
        key:     PRNGKey
        state:   (*D) current state of the environment
        action:  (*K) action to apply to state and environment

        Returns
        ------- 
        reward:      The sampled reward associated with the reached state and action
        next_state:  The next state of the environment
        """
        next_state = self.transition(key, state, action)
        reward = self._reward(state)
        return next_state, reward


    def _reward(self, state: Array):
        """Normalised pendulum height """
        theta = state[2]
        return 0.5 * (1.0 - jnp.cos(theta))


    def transition(self, key: PRNGKey, state: Array, action: Array):
        """Apply one action over several smaller physics steps."""
        keys = jr.split(key, self.physics_steps)

        def body(i, current_state):
            return self._step(keys[i], current_state, action)
        return lax.fori_loop(0, self.physics_steps, body, state)


    def _step(self, key: PRNGKey, state: Array, action: Array):
        """Apply one stochastic physics step."""
        x, x_dot, theta, theta_dot = state

        action = jnp.squeeze(jnp.asarray(action))
        action = jnp.clip(action, -1.0, 1.0)
        force = self.force_scale * action

        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)

        q = self.gravity * sin_theta
        q += self.hinge_damping * theta_dot / (self.pole_mass * self.pole_length)

        denominator = self.cart_mass + self.pole_mass - self.pole_mass * cos_theta**2

        numerator = force - self.cart_friction * x_dot
        numerator += self.pole_mass * self.pole_length * theta_dot**2 * sin_theta
        numerator += self.pole_mass * cos_theta * q

        x_ddot = numerator / denominator
        theta_ddot = -(x_ddot * cos_theta + q) / self.pole_length

        # semi-implicit Euler integration.
        x_dot = x_dot + self.dt * x_ddot
        theta_dot = theta_dot + self.dt * theta_ddot
        x = x + self.dt * x_dot
        theta = angle_normalise(theta + self.dt * theta_dot)

        next_state = jnp.stack([x, x_dot, theta, theta_dot])

        # add noise to state
        noise = jnp.sqrt(self.dt) * self.process_std * jr.normal(key, (4,))
        next_state = next_state + noise
        next_state = next_state.at[2].set(angle_normalise(next_state[2]))

        return next_state
    

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

    state_key, obs_key, key_C, key_d = jr.split(key, 4)

    J, N, T = num_factors, num_sequences, num_timesteps
    M = N + N // 4      # 25% extra for validation data

    # initialise
    initial_state = jnp.array([
        0.0,  # cart position
        0.0,  # cart velocity
        0.0,  # downward pendulum angle
        0.0,  # angular velocity
    ])
    environment = PendulumEnvironment(T=T)

    # sample latent states using policy
    _, latent_sample, actions = environment.sample(
        state_key,
        initial_state,
        policy,
        num_samples=M,
    )

    # sample observations
    D = latent_sample.shape[-1]
    emission_params = {
        "C": jr.normal(key_C, shape=(J, D, D)),
        "d": jr.normal(key_d, shape=(J, D)),
        "R": jnp.broadcast_to(emission_cov * jnp.eye(D), (J, D, D)),
    }

    def sample_single_factor(C, d, R, key):
        means = latent_sample @ C.T + d
        noise = jr.multivariate_normal(key, jnp.zeros(D), R, shape=(M, T))
        return means + noise

    obs_samples = vmap(sample_single_factor)(
        emission_params["C"],
        emission_params["d"],
        emission_params["R"],
        jr.split(obs_key, J),
    )

    # TODO make params look like this - use Q in environment.sample
    params = {
            'm1': np.zeros(K),
            'Q1': np.eye(K),
            
            'A': np.eye(K), # no dynamics (random walk)
            'b': np.zeros(K),
            'Q': np.eye(K), # independent noise
            'R': np.eye(D) * np.sqrt(emission_cov) # keep R constant across J for now
        }
    
    return Dataset(
        train_data=tuple(o[:N] for o in obs_samples),
        train_states=latent_sample[:N],
        val_data=tuple(o[N:] for o in obs_samples),
        val_states=latent_sample[N:],
        params={
            **emission_params,
            "train_actions": actions[:N],
            "val_actions": actions[N:],
        },
    )