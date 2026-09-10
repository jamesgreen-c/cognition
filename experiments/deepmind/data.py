import jax.numpy as jnp
import numpy as np

from jax import Array
from dm_control import suite
from rp_slac.environment import MuJoCoSimulationEnvironment


class DmControlEnvironment(MuJoCoSimulationEnvironment):

    def __init__(
            self,
            domain_name: str,
            task_name: str,
            num_buffers: int,
            action_repeat: int = 1,
            width: int = 64,
            height: int = 64,
            camera_id: int = 0,
        ):
        super().__init__(
            num_buffers=num_buffers,
            action_repeat=action_repeat,
        )

        self.domain_name = domain_name
        self.task_name = task_name
        self.width = width
        self.height = height
        self.camera_id = camera_id

        # obtain the action specification without retaining another simulator.
        spec_env = self._make_env(seed=0)
        spec = spec_env.action_spec()
        self.action_lower = jnp.asarray(spec.minimum)
        self.action_upper = jnp.asarray(spec.maximum)

    def _make_env(self, seed: int):
        return suite.load(
            domain_name=self.domain_name,
            task_name=self.task_name,
            task_kwargs={"random": seed},
        )

    def _state(self, env) -> Array:
        return jnp.asarray(env.physics.get_state())

    def _observation(self, env) -> Array:
        pixels = env.physics.render(
            width=self.width,
            height=self.height,
            camera_id=self.camera_id,
        )
        return jnp.asarray(pixels)

    def _step_simulator(self, env, action: Array):
        time_step = env.step(np.asarray(action))
        reward = float(time_step.reward or 0.0)
        # reward = 0.0 if time_step.reward is None else float(time_step.reward)
        terminal = bool(time_step.last())
        return reward, terminal

    def _action_bounds(self):
        return self.action_lower, self.action_upper

    # override preprocess
    def preprocess_observation(self, observation: Array) -> Array:
        return observation.astype(jnp.float32) / 255.0