import jax.numpy as jnp
import mujoco

from mujoco import mjx
from mujoco_playground import dm_control_suite as suite

from rp_slac.environment import MuJoCoSimulationEnvironment


_ENV_NAMES = {
    ("acrobot", "swingup"): "AcrobotSwingup",
    ("ball_in_cup", "catch"): "BallInCup",
    ("cartpole", "balance"): "CartpoleBalance",
    ("cartpole", "balance_sparse"): "CartpoleBalanceSparse",
    ("cartpole", "swingup"): "CartpoleSwingup",
    ("cartpole", "swingup_sparse"): "CartpoleSwingupSparse",
    ("cheetah", "run"): "CheetahRun",
    ("finger", "spin"): "FingerSpin",
    ("finger", "turn_easy"): "FingerTurnEasy",
    ("finger", "turn_hard"): "FingerTurnHard",
    ("fish", "swim"): "FishSwim",
    ("hopper", "hop"): "HopperHop",
    ("hopper", "stand"): "HopperStand",
    ("humanoid", "stand"): "HumanoidStand",
    ("humanoid", "walk"): "HumanoidWalk",
    ("humanoid", "run"): "HumanoidRun",
    ("pendulum", "swingup"): "PendulumSwingup",
    ("point_mass", "easy"): "PointMass",
    ("reacher", "easy"): "ReacherEasy",
    ("reacher", "hard"): "ReacherHard",
    ("swimmer", "swimmer6"): "SwimmerSwimmer6",
    ("walker", "run"): "WalkerRun",
    ("walker", "stand"): "WalkerStand",
    ("walker", "walk"): "WalkerWalk",
}


_DEFAULT_CAMERAS = {
    "AcrobotSwingup": "fixed",
    "BallInCup": "cam0",
    "CartpoleBalance": "fixed",
    "CartpoleBalanceSparse": "fixed",
    "CartpoleSwingup": "fixed",
    "CartpoleSwingupSparse": "fixed",
    "CheetahRun": "side",
    "FingerSpin": "cam0",
    "FingerTurnEasy": "cam0",
    "FingerTurnHard": "cam0",
    "FishSwim": "fixed_top",
    "HopperHop": "cam0",
    "HopperStand": "cam0",
    "HumanoidStand": "side",
    "HumanoidWalk": "side",
    "HumanoidRun": "side",
    "PendulumSwingup": "fixed",
    "PointMass": "cam0",
    "ReacherEasy": "fixed",
    "ReacherHard": "fixed",
    "SwimmerSwimmer6": "tracking1",
    "WalkerRun": "side",
    "WalkerStand": "side",
    "WalkerWalk": "side",
}


class DmControlEnvironment(MuJoCoSimulationEnvironment):

    def __init__(
            self,
            domain_name: str,
            task_name: str,
            num_buffers: int,
            actor_history: int,
            action_repeat: int = 1,
            episode_length: int = 1000,
            width: int = 64,
            height: int = 64,
            rgb: bool = False,
            camera_id: int | str | None = None,
        ):
        key = (domain_name, task_name)

        if key not in _ENV_NAMES:
            raise ValueError(f"Unsupported DM Control environment: {key}.")

        self.domain_name = domain_name
        self.task_name = task_name
        self.env_name = _ENV_NAMES[key]
        self.width = width
        self.height = height
        self.rgb = rgb
        self.camera = _DEFAULT_CAMERAS[self.env_name] if camera_id is None else camera_id

        super().__init__(
            num_buffers=num_buffers, 
            action_repeat=action_repeat, 
            actor_history=actor_history, 
            episode_length=episode_length
        )

        ctrl_range = self.env.mj_model.actuator_ctrlrange
        self.action_lower = jnp.asarray(ctrl_range[:, 0])
        self.action_upper = jnp.asarray(ctrl_range[:, 1])

    def _make_env(self):
        config = suite.get_default_config(self.env_name)

        config.vision = False
        config.impl = "warp"
        config.action_repeat = 1

        return suite.load(self.env_name, config=config)
        
    def _setup_observation(self):

        if isinstance(self.camera, str):
            camera_id = mujoco.mj_name2id(self.env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
        else:
            camera_id = int(self.camera)

        if camera_id < 0 or camera_id >= self.env.mj_model.ncam:
            available = [
                mujoco.mj_id2name(self.env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, idx)
                for idx in range(self.env.mj_model.ncam)
            ]
            raise ValueError(
                f"Camera '{self.camera}' was not found for {self.env_name}. Available cameras: {available}.")

        self.camera_id = camera_id

        # enable only the selected camera
        render_rgb = [False] * self.env.mj_model.ncam
        render_rgb[self.camera_id] = True

        self.render_context = mjx.create_render_context(
            mjm=self.env.mj_model,
            nworld=self.num_buffers,
            cam_res=(self.width, self.height),
            use_textures=True,
            use_shadows=True,
            render_rgb=render_rgb,
            render_depth=[False] * self.env.mj_model.ncam,
            enabled_geom_groups=[0, 1, 2],
        )

        # keep render_context alive and pass only its lightweight pytree handle through compiled computations.
        self.render_context_pytree = self.render_context.pytree()

    def _observe(self, state):
        data = mjx.refit_bvh(self.env.mjx_model, state.data, self.render_context_pytree)
        packed_rgb, _, data = mjx.render(self.env.mjx_model, data, self.render_context_pytree)
        observation = mjx.get_rgb(self.render_context_pytree, self.camera_id, packed_rgb)
        observation = observation[..., :3].astype(jnp.float32)      # MJX is already returning images in [0,1] so no need to divide / 255.0 

        if not self.rgb:
            weights = jnp.asarray([0.2989, 0.5870, 0.1140])
            observation = jnp.sum(observation * weights, axis=-1, keepdims=True)

        state = state.replace(data=data) # , obs=observation)
        return state, observation

    def _action_bounds(self):
        return self.action_lower, self.action_upper


if __name__ == "__main__":
    from pathlib import Path

    import jax
    import numpy as np
    from PIL import Image

    env = DmControlEnvironment(
        domain_name="cheetah",
        task_name="run",
        num_buffers=32,
        actor_history=1,
        width=64,
        height=64,
    )

    carry = jax.jit(lambda key: env.initial_carry(key, 32))(jax.random.key(0))
    observation = np.asarray(env.current_observation(carry))

    print(observation.shape, observation.dtype)
    print(observation.min(), observation.max())

    # Accept (batch, height, width, channels) or (batch, history, height, width, channels).
    if observation.ndim == 4:
        observation = observation[:, None]
    if observation.ndim != 5 or observation.shape[-1] not in (1, 3):
        raise ValueError(f"Unexpected observation shape: {observation.shape}")

    output_dir = Path("images")
    output_dir.mkdir(exist_ok=True)

    images = []
    for buffer_idx, history in enumerate(observation):
        for history_idx, frame in enumerate(history):
            pixels = np.uint8(np.round(np.clip(frame, 0.0, 1.0) * 255))
            if pixels.shape[-1] == 1:
                pixels = pixels[..., 0]

            image = Image.fromarray(pixels)
            image.save(output_dir / f"buffer_{buffer_idx:02d}_history_{history_idx:02d}.png")
            images.append(image.convert("RGB"))

    columns = 8
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * env.width, rows * env.height))
    for idx, image in enumerate(images):
        sheet.paste(image, ((idx % columns) * env.width, (idx // columns) * env.height))

    sheet.save(output_dir / "contact_sheet.png")
    print(f"Saved {len(images)} images and a contact sheet to {output_dir.resolve()}")

# if __name__ == "__main__":
#     import jax

#     env = DmControlEnvironment(
#         domain_name="cheetah",
#         task_name="run",
#         num_buffers=32,
#         actor_history=1,
#         width=64,
#         height=64,
#     )

#     carry = jax.jit(lambda key: env.initial_carry(key, 32))(jax.random.key(0))
#     observation = env.current_observation(carry)

#     print(observation.shape)
#     print(observation.dtype)
#     print(observation.min(), observation.max())

