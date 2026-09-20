import optax

import jax
import jax.random as jr
import jax.numpy as jnp
import numpy as np

from tqdm import tqdm
from typing import Callable

from jax import Array, vmap
from jax.random import PRNGKey
from jax.tree_util import tree_map

from rp_slac.environment import JAXEnvironment, MuJoCoSimulationEnvironment
from rp_slac.free_energy.model_fe import ConstrainedIVFreeEnergy
from rp_slac.free_energy.control_fe import ControlFreeEnergy

from rp_slac.config import Config
from rp_slac.distributions import AllParams
from rp_slac.utils.math import scale_sv, clip_sv


EPS = 1e-3


class RPSLAC:
    params: AllParams
    opt_states: list[optax.OptState]
    opts: list[optax.GradientTransformation]
    itr: int

    def __init__(
            self,
            model: ConstrainedIVFreeEnergy,
            control: ControlFreeEnergy, 
            environment: JAXEnvironment | MuJoCoSimulationEnvironment,
            config: Config,
            logger: Callable = lambda *x: {}
    ):
        
        self.model = model
        self.control = control
        self.env = environment
        self.config = config
        self.logger = logger
        self.itr = 0

        assert self.config.actor_state in ("observation", "latent")

        if (self.config.actor_state == "observation" and self.config.actor_history != self.env.actor_history):
            raise ValueError(
                "config.actor_history must match environment.actor_history "
                f"({self.config.actor_history} != {self.env.actor_history})."
            )
        if self.config.initial_steps < 1:
            raise ValueError("initial_steps must be positive.")
        if self.config.initial_steps < self.config.sequence_length:
            raise ValueError("initial_steps must be at least sequence_length.")
        if self.config.initial_steps > self.config.capacity:
            raise ValueError("initial_steps cannot exceed replay capacity.")

    def pretrain_step(self, itr: int, params: AllParams, opt_states: dict[optax.OptState], data: tuple[Array]):
            model_params, model_opt_states, model_loss, _ = self._update_model(itr, params, opt_states, data)
            new_params = {**params, **model_params}
            new_opt_states = {**opt_states, **model_opt_states}
            return new_params, new_opt_states, model_loss

    def train_step(
            self,
            key: PRNGKey,
            itr: int,
            params: AllParams,
            opt_states: dict[optax.OptState],
            data: tuple[Array],
    ) -> tuple[dict, dict[optax.OptState], dict[float]]: 
        actor_key, critic_key, sampling_key = jr.split(key, 3)

        # calculate all updates from the previous step parameters
        model_params, model_opt_states, model_loss, filter_posterior = self._update_model(itr,
                                                                                            params,
                                                                                            opt_states,
                                                                                            data)
        # shape = filter_posterior.params["mean"].shape[:-1]
        latents = filter_posterior.sample(sampling_key, shape=None)   # filter_posterior.mean

        critic_params, critic_opt_state, critic_loss = self._update_critic(critic_key, params, opt_states, latents, data)
        actor_params, actor_opt_state, actor_loss, aux = self._update_actor(actor_key, params, opt_states, latents, data)
        log_alpha, alpha_opt_state, alpha_loss, entropy_gap = self._update_alpha(params, opt_states, aux["log_probs"])

        # combine the independently updated parameter subsets
        new_params = {
            **params,
            **model_params,
            "critic": critic_params,
            "actor": actor_params,
            "log_alpha": log_alpha
        }
        new_opt_states = {
            **opt_states,
            **model_opt_states,
            "critic": critic_opt_state,
            "actor": actor_opt_state,
            "alpha": alpha_opt_state
        }

        aux = {**aux, "entropy_gap": entropy_gap}
        losses = {"model": model_loss, "critic": critic_loss, "actor": actor_loss, "alpha": alpha_loss}
        return new_params, new_opt_states, losses, aux
    

    def fit(self, use_pbar: bool = True) -> None:
        """ 
        
        Parameters 
        ----------
        replay_buffer:  Pregenerated data from a random policy 
        """
        key, buffer_key, init_key = jr.split(jr.PRNGKey(self.config.seed), 3)

        replay_buffer, env_states = self._init_replay_buffer(buffer_key, use_pbar)
        key, init_batch_key = jr.split(key)
        init_batch = self._get_batch(init_batch_key, replay_buffer)
        self.params, self.opt_states, self.opts = self.init(init_key, init_batch)

        # The environment and updates can be JIT compiled; the replay buffer stays on the CPU.
        compile_environment = self.config.jit and self.env.jax_compatible
        policy_step = jax.jit(self.policy_step) if compile_environment else self.policy_step
        collect_step = jax.jit(self.env.collect_step) if compile_environment else self.env.collect_step
        pretrain_step = jax.jit(self.pretrain_step) if self.config.jit else self.pretrain_step
        train_step = jax.jit(self.train_step) if self.config.jit else self.train_step

        # run model pretraining
        self.pretraining_losses = []

        pbar = tqdm(range(self.config.num_pretrain), disable=not(use_pbar), desc="Pretraining")
        for self.pretrain_itr in pbar:
            key, subkey = jr.split(key)
            batch = self._get_batch(subkey, replay_buffer)

            self.params, self.opt_states, loss = pretrain_step(self.pretrain_itr, self.params, self.opt_states, batch)
            self._stabilise_params()
            
            self.pretraining_losses.append(loss)
            pbar.set_postfix(loss=float(loss))

        # run training
        self.model_losses = []
        self.critic_losses = []
        self.actor_losses = []
        self.alpha_losses = []

        self.average_rewards = []
        self.actor_stats = []
        self.alpha_hist = []

        pbar = tqdm(range(self.config.num_iter), disable=not(use_pbar), desc="Training")
        for self.itr in pbar:
            key, policy_key, env_key, batch_key, train_key = jr.split(key, 5)

            actor_observation = self.env.actor_observation(env_states)
            actions, log_probs = policy_step(policy_key, self.params["actor"], actor_observation)
            env_states, new_obs, rewards, flags = collect_step(env_key, env_states, actions)
            replay_buffer = self._append_experience(replay_buffer, new_obs, actions, rewards, flags, log_probs)
            batch = self._get_batch(batch_key, replay_buffer)

            self.params, self.opt_states, losses, aux = train_step(train_key, 
                                                                   self.itr, 
                                                                   self.params, 
                                                                   self.opt_states, 
                                                                   batch)
            self._stabilise_params()

            # calculate average reward
            size = int(replay_buffer["size"])
            average_reward = replay_buffer["data"][2][:, :size].mean()

            self.model_losses.append(losses["model"])
            self.critic_losses.append(losses["critic"])
            self.actor_losses.append(losses["actor"])
            self.alpha_losses.append(losses["alpha"])

            self.average_rewards.append(average_reward)
            self.actor_stats.append(aux)
            self.alpha_hist.append(float(self.params["log_alpha"]))

            pbar.set_postfix(
                average_reward="{:.4f}".format(float(average_reward)),
                model_loss="{:.3f}".format(float(losses["model"])),
                critic_loss="{:.3f}".format(float(losses["critic"])),
                actor_loss="{:.3f}".format(float(losses["actor"])),
                alpha_loss="{:.3f}".format(float(losses["alpha"]))
            )

        return self.params, replay_buffer, env_states, key

    def train_continue(
            self,
            replay_buffer: dict,
            env_states,
            params: AllParams,
            opts: dict,
            start_itr: int,
            opt_states: dict,
            key: PRNGKey,
            use_pbar: bool = True,
    ):
        """Run config.num_iter further online training steps from a checkpoint."""
        if start_itr < 0 or self.config.num_iter < 1:
            raise ValueError("start_itr must be nonnegative and num_iter must be positive.")

        self.model.configure(self.config)
        self.control.configure(self.config, replay_buffer["data"])

        self.params = params
        self.opt_states = opt_states
        self.opts = opts

        for name in (
            "model_losses", "critic_losses", "actor_losses", "alpha_losses",
            "average_rewards", "actor_stats", "alpha_hist",
        ):
            if not hasattr(self, name):
                setattr(self, name, [])

        compile_environment = self.config.jit and self.env.jax_compatible
        policy_step = jax.jit(self.policy_step) if compile_environment else self.policy_step
        collect_step = jax.jit(self.env.collect_step) if compile_environment else self.env.collect_step
        train_step = jax.jit(self.train_step) if self.config.jit else self.train_step

        end_itr = start_itr + self.config.num_iter
        pbar = tqdm(range(start_itr, end_itr), disable=not use_pbar, desc="Training")
        for self.itr in pbar:
            key, policy_key, env_key, batch_key, train_key = jr.split(key, 5)

            actor_observation = self.env.actor_observation(env_states)
            actions, log_probs = policy_step(policy_key, self.params["actor"], actor_observation)
            env_states, new_obs, rewards, flags = collect_step(env_key, env_states, actions)
            replay_buffer = self._append_experience(replay_buffer, new_obs, actions, rewards, flags, log_probs)
            batch = self._get_batch(batch_key, replay_buffer)

            self.params, self.opt_states, losses, aux = train_step(train_key, 
                                                                   self.itr, 
                                                                   self.params, 
                                                                   self.opt_states, 
                                                                   batch)
            self._stabilise_params()

            size = int(replay_buffer["size"])
            average_reward = replay_buffer["data"][2][:, :size].mean()

            self.model_losses.append(losses["model"])
            self.critic_losses.append(losses["critic"])
            self.actor_losses.append(losses["actor"])
            self.alpha_losses.append(losses["alpha"])
            self.average_rewards.append(average_reward)
            self.actor_stats.append(aux)
            self.alpha_hist.append(float(self.params["log_alpha"]))

            pbar.set_postfix(
                average_reward="{:.4f}".format(float(average_reward)),
                model_loss="{:.3f}".format(float(losses["model"])),
                critic_loss="{:.3f}".format(float(losses["critic"])),
                actor_loss="{:.3f}".format(float(losses["actor"])),
                alpha_loss="{:.3f}".format(float(losses["alpha"])),
            )

        self.itr = end_itr
        return self.params, replay_buffer, env_states, key
    
    def init(self, key: PRNGKey, initial_batch: tuple[Array]):
        """ initialise all agent parameters """
        model_key, control_key = jr.split(key)

        # control initialisation
        control_params, control_opt_states, control_opts = self.control.init(
            control_key, 
            initial_batch,
            self.config,
            self.model.model.latent_dim
        )

        # add rpm initialisation
        model_params, model_opt_states, model_opts = self.model.init(
            model_key, 
            initial_batch,
            self.config
        )

        params = {**control_params, **model_params}
        opt_states = {**control_opt_states, **model_opt_states}
        opts = {**control_opts, **model_opts}

        return params, opt_states, opts
    
    def _init_replay_buffer(self, key: PRNGKey, use_pbar: bool):
        N = self.config.num_buffers
        I = self.config.initial_steps
        C = self.config.capacity

        # compile required functions
        compile_environment = self.config.jit and self.env.jax_compatible
        random_policy_step = jax.jit(self.random_policy_step) if compile_environment else self.random_policy_step
        collect_step = jax.jit(self.env.collect_step) if compile_environment else self.env.collect_step
        initial_carry = (
            jax.jit(lambda init_key: self.env.initial_carry(init_key, N))
            if compile_environment
            else lambda init_key: self.env.initial_carry(init_key, N)
        )

        key, initial_key = jr.split(key)
        env_states = initial_carry(initial_key)
        initial_observation = self.env.current_observation(env_states)

        # get shapes and dtypes from a single transition step
        key, policy_key, env_key = jr.split(key, 3)
        actions, log_probs = random_policy_step(policy_key, initial_observation)
        env_states, next_observation, rewards, flags = collect_step(
            env_key,
            env_states,
            actions,
        )

        # store image observations as uint8 when the renderer provides floats in [0, 1].
        initial_observation = np.asarray(jax.device_get(initial_observation))
        quantize_observations = (
            initial_observation.ndim >= 4
            and np.issubdtype(initial_observation.dtype, np.floating)
            and np.isfinite(initial_observation).all()
            and initial_observation.min() >= 0.0
            and initial_observation.max() <= 1.0
        )
        if quantize_observations:
            initial_observation = np.rint(initial_observation * 255.0).astype(np.uint8)

        observation_buffer = np.empty((N, C + 1) + initial_observation.shape[1:], dtype=initial_observation.dtype)
        observation_buffer[:, 0] = initial_observation
        action_buffer = np.empty((N, C) + actions.shape[1:], dtype=actions.dtype)
        reward_buffer = np.empty((N, C), dtype=rewards.dtype)
        flag_buffer = np.empty((N, C), dtype=flags.dtype)
        log_prob_buffer = np.empty((N, C), dtype=log_probs.dtype)

        replay_buffer = {
            "data": (observation_buffer, action_buffer, reward_buffer, flag_buffer, log_prob_buffer),
            "size": 0,
            "write": 0,              # next transition slot
            "obs_start": 0,          # oldest observation slot
            "quantized_obs": quantize_observations,
        }
        replay_buffer = self._append_experience(replay_buffer, next_observation, actions, rewards, flags, log_probs)

        # run random experience collection
        pbar = tqdm(range(1, I), disable=not(use_pbar), desc="Getting random experience")
        for _ in pbar:
            key, policy_key, env_key = jr.split(key, 3)
            observation = self.env.current_observation(env_states)
            actions, log_probs = random_policy_step(policy_key, observation)
            env_states, next_observation, rewards, flags = collect_step(env_key, env_states, actions)
            replay_buffer = self._append_experience(replay_buffer, next_observation, actions, rewards, flags, log_probs)

        return replay_buffer, env_states

    def _get_batch(self, key: PRNGKey, replay_buffer: dict):
        """
        Sample B contiguous windows independently from each of N buffers,
        without crossing episode boundaries.

        Returns
        -------
        observations: (N * B, tau + 1, ...)
        actions:      (N * B, tau, ...)
        rewards:      (N * B, tau)
        discounts:    (N * B, tau)
        """
        observations, actions, rewards, discounts, log_probs = replay_buffer["data"]
        size = replay_buffer["size"]

        N = self.config.num_buffers
        B = self.config.batch_size
        tau = self.config.sequence_length
        C = self.config.capacity

        if size < tau:
            raise ValueError("The replay buffer has fewer transitions than sequence_length.")

        # derive the host RNG from the supplied JAX key so batching remains reproducible.
        rng = np.random.default_rng(np.asarray(jax.device_get(jr.key_data(key))))
        transition_start = (replay_buffer["write"] - size) % C
        ordered_indices = (transition_start + np.arange(size)) % C
        starts = np.empty((N, B), dtype=np.intp)

        for n in range(N):
            terminals = discounts[n, ordered_indices] != 1
            cumulative = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(terminals, dtype=np.int32)))
            valid_starts = np.flatnonzero(cumulative[tau:] == cumulative[:-tau])

            if valid_starts.size == 0:
                raise ValueError(f"No episode-contiguous replay windows for buffer {n}.")
            starts[n] = rng.choice(valid_starts, size=B, replace=True)

        buffer_indices = np.arange(N)[:, None, None]
        transition_indices = (transition_start + starts[..., None] + np.arange(tau)) % C
        observation_indices = (replay_buffer["obs_start"] + starts[..., None] + np.arange(tau + 1)) % (C + 1)
        batch = (
            observations[buffer_indices, observation_indices],
            actions[buffer_indices, transition_indices],
            rewards[buffer_indices, transition_indices],
            discounts[buffer_indices, transition_indices],
            log_probs[buffer_indices, transition_indices],
        )
        batch = tuple(x.reshape((N * B,) + x.shape[2:]) for x in batch)

        observations, actions, rewards, discounts, log_probs = (jnp.asarray(x) for x in batch)
        if replay_buffer["quantized_obs"]:
            observations = observations.astype(jnp.float32) / 255.0
        observations = self.env.preprocess_observation(observations)

        return observations, actions, rewards, discounts, log_probs

    def _append_experience(
            self,
            replay_buffer: dict,
            new_obs: Array,
            new_act: Array,
            new_rew: Array,
            new_flags: Array,
            new_log_ps: Array,
        ):
        C = self.config.capacity
        size = replay_buffer["size"]
        write = replay_buffer["write"]
        obs_start = replay_buffer["obs_start"]
        obs, act, rew, flags, log_ps = replay_buffer["data"]

        new_obs, new_act, new_rew, new_flags, new_log_ps = (
            np.asarray(x) for x in jax.device_get((new_obs, new_act, new_rew, new_flags, new_log_ps))
        )
        if replay_buffer["quantized_obs"]:
            new_obs = np.rint(np.clip(new_obs, 0.0, 1.0) * 255.0).astype(np.uint8)

        act[:, write] = new_act
        rew[:, write] = new_rew
        flags[:, write] = new_flags
        log_ps[:, write] = new_log_ps
        obs[:, (obs_start + size + 1) % (C + 1)] = new_obs

        if size == C:
            replay_buffer["obs_start"] = (obs_start + 1) % (C + 1)
        else:
            replay_buffer["size"] = size + 1
        replay_buffer["write"] = (write + 1) % C
        return replay_buffer


    def _update_model(self, itr: int, params, opt_states, data):
        """ Update RPM model parameters using Kalman smoothing """

        beta = self.config.beta_schedule(itr)
        em = self.config.em

        obs = data[0]
        actions = data[1]

        _params = {"prior": params["prior"], "rpm": params["rpm"]}
        (loss, aux), grads = jax.value_and_grad(self.model.loss, has_aux=True)(_params, obs, actions, beta, em)

        new_params = {}
        new_opt_states = {}
        for name in ("prior", "rpm"):
            updates, new_opt_states[name] = self.opts[name].update(grads[name], opt_states[name], params[name])
            new_params[name] = optax.apply_updates(params[name], updates)
        return new_params, new_opt_states, loss, aux["filter_posterior"]


    def _update_critic(self, key, params, opt_states, latents, data):
        """ Update soft critic parameters """
        rho = self.config.target_update_rate
        _params = {"actor": params["actor"], "critic": params["critic"], "log_alpha": params["log_alpha"]}

        # latest param updates
        loss, grads = jax.value_and_grad(self.control.critic_loss, argnums=1)(key, _params, latents, data)
        latest_updates, new_critic_opt_states = self.opts["critic"].update(grads["critic"]["latest"], 
                                                                           opt_states["critic"], 
                                                                           params["critic"]["latest"])
        latest_params = optax.apply_updates(params["critic"]["latest"], latest_updates)

        # target param updates
        target_update = lambda _tps, _lps: (1.0 - rho) * _tps + rho * _lps
        target_params = tree_map(target_update, params["critic"]["target"], latest_params)

        critic_params = {"latest": latest_params, "target": target_params}
        return critic_params, new_critic_opt_states, loss


    def _update_actor(self, key, params, opt_states, latents, data):
        """ Update soft actor parameters """

        _params = {"actor": params["actor"], "critic": params["critic"], "log_alpha": params["log_alpha"]}
        (loss, aux), grads = jax.value_and_grad(
            self.control.actor_loss, argnums=1, has_aux=True
        )(key, _params, latents, data)

        updates, new_actor_opt_states = self.opts["actor"].update(grads["actor"], opt_states["actor"], params["actor"])
        actor_params = optax.apply_updates(params["actor"], updates)
        return actor_params, new_actor_opt_states, loss, aux


    def _update_alpha(self, params, opt_states, log_probs: Array):
        """ Update log temperature parameter """
        _params = {"log_alpha": params["log_alpha"]}
        (loss, aux), grads = jax.value_and_grad(self.control.alpha_loss, has_aux=True)(_params, log_probs)

        updates, alpha_opt_state = self.opts["alpha"].update(grads["log_alpha"], opt_states["alpha"], params["log_alpha"])
        log_alpha = optax.apply_updates(params["log_alpha"], updates)
        return log_alpha, alpha_opt_state, loss, aux["entropy_gap"]
    

    def _stabilise_params(self):
        # TODO stabilisation would require new Q calculation so needs to go into prior? 
        if self.config.stabilise_A == 'scale':
            self.params["prior"]["A"] = scale_sv(self.params["prior"]["A"], EPS)
        elif self.config.stabilise_A == 'clip':
            self.params["prior"]["A"] = clip_sv(self.params["prior"]["A"], EPS)


    def apply(self, key: PRNGKey, params: dict, observations: Array, deterministic: bool = True) -> Array:
        """ 
        Get the action for the given observation and parameters
        
        Parameters
        ----------
        key:           RNG
        params:        Trained model and control parameters
        obseravtions:  (B, D) or (D,) for a set of B (or one) observations 
        """
        observations = self.env.preprocess_observation(observations)

        if observations.ndim == 1:
            observations = observations[None]

        if deterministic:
            return vmap(lambda state: self.control.mean_policy(params, state))(observations)

        keys = jr.split(key, observations.shape[0])
        return vmap(lambda _key, state: self.control.policy(_key, params, state))(keys, observations)

    def policy_step(self, key: PRNGKey, actor_params: dict, actor_observation: Array):
        actor_observation = self.env.preprocess_observation(actor_observation)
        keys = jr.split(key, self.config.num_buffers)
        return self.control.vmapped_actor(keys, actor_params, actor_observation)

    def random_policy_step(self, key: PRNGKey, observation: Array):
        """Sample one independent random action for each replay buffer."""
        keys = jr.split(key, self.config.num_buffers)
        return vmap(self.env.random_action)(keys, observation)
