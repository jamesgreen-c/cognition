import optax

import jax
import jax.random as jr
import jax.numpy as jnp

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
        actor_key, critic_key = jr.split(key)

        # calculate all updates from the previous step parameters
        model_params, model_opt_states, model_loss, filtered_means = self._update_model(itr, 
                                                                                          params, 
                                                                                          opt_states, 
                                                                                          data)
        latents = filtered_means   # TODO: posterior.sample(sampling_key)

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
        self.params, self.opt_states, self.opts = self.init(init_key, replay_buffer["data"])

        # JIT compile (or don't) policy, experience, pretraining and training functions 
        compile_environment = self.config.jit and self.env.jax_compatible
        policy_step = jax.jit(self.policy_step) if compile_environment else self.policy_step
        collect_step = jax.jit(self.env.collect_step) if compile_environment else self.env.collect_step
        append_experience = jax.jit(self._append_experience) if compile_environment else self._append_experience
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
            replay_buffer = append_experience(replay_buffer, new_obs, actions, rewards, flags, log_probs)
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

        return self.params, replay_buffer
    
    def init(self, key: PRNGKey, replay_buffer: tuple[Array]):
        """ initialise all agent parameters """
        model_key, control_key = jr.split(key)

        # control initialisation
        control_params, control_opt_states, control_opts = self.control.init(
            control_key, 
            replay_buffer,
            self.config,
            self.model.model.latent_dim
        )

        # add rpm initialisation
        model_params, model_opt_states, model_opts = self.model.init(
            model_key, 
            replay_buffer, 
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
        append_experience = jax.jit(self._append_experience) if compile_environment else self._append_experience
        initial_carry = (
            jax.jit(lambda init_key: self.env.initial_carry(init_key, N))
            if compile_environment
            else lambda init_key: self.env.initial_carry(init_key, N)
        )

        key, initial_key = jr.split(key)
        env_states = initial_carry(initial_key)
        initial_observation = self.env.current_observation(env_states)

        # Get shapes and dtypes from a single transition step
        key, policy_key, env_key = jr.split(key, 3)
        actions, log_probs = random_policy_step(policy_key, initial_observation)
        env_states, next_observation, rewards, flags = collect_step(
            env_key,
            env_states,
            actions,
        )

        # initialise buffer
        observation_buffer = jnp.zeros((N, C + 1) + initial_observation.shape[1:]).at[:, 0].set(initial_observation)
        action_buffer = jnp.zeros((N, C) + actions.shape[1:], dtype=actions.dtype)
        reward_buffer = jnp.zeros((N, C), dtype=rewards.dtype)
        flag_buffer = jnp.zeros((N, C), dtype=flags.dtype)
        log_prob_buffer = jnp.zeros((N, C), dtype=log_probs.dtype)

        replay_buffer = {
            "data": (observation_buffer, action_buffer, reward_buffer, flag_buffer, log_prob_buffer),
            "size": jnp.asarray(0, dtype=jnp.int32),
        }
        replay_buffer = append_experience(replay_buffer, next_observation, actions, rewards, flags, log_probs)

        # run random experience collection
        pbar = tqdm(range(1, I), disable=not(use_pbar), desc="Getting random experience")
        for _ in pbar:
            key, policy_key, env_key = jr.split(key, 3)
            observation = self.env.current_observation(env_states)
            actions, log_probs = random_policy_step(policy_key, observation)
            env_states, next_observation, rewards, flags = collect_step(env_key, env_states, actions)
            replay_buffer = append_experience(replay_buffer, next_observation, actions, rewards, flags, log_probs)

        jax.block_until_ready(replay_buffer["size"])
        return replay_buffer, env_states

    def _get_batch(self, key: PRNGKey, replay_buffer: tuple[Array]):
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
        capacity = self.config.capacity
        num_starts = capacity - tau + 1

        keys = jr.split(key, N)

        def sample_buffer(key, obs, act, rew, disc, log_ps):

            # number of terminal transitions in every length-tau window
            terminals = 1.0 - disc
            cumulative = jnp.concatenate([jnp.zeros((1,), dtype=terminals.dtype), jnp.cumsum(terminals)])
            terminal_counts = cumulative[tau:] - cumulative[:-tau]

            # valid only when the complete window belongs to one episode and we arent an empty buffer region
            starts = jnp.arange(num_starts)
            within_filled_buffer = starts + tau <= size
            within_one_episode = terminal_counts == 0
            valid_starts = within_filled_buffer & within_one_episode

            # uniform sampling with replacement over valid starts
            logits = jnp.where(valid_starts, 0.0, -jnp.inf)
            starts = jr.categorical(key, logits, shape=(B,))

            def sample_window(start):
                return (
                    jax.lax.dynamic_slice_in_dim(obs, start, tau + 1, axis=0),
                    jax.lax.dynamic_slice_in_dim(act, start, tau, axis=0),
                    jax.lax.dynamic_slice_in_dim(rew, start, tau, axis=0),
                    jax.lax.dynamic_slice_in_dim(disc, start, tau, axis=0),
                    jax.lax.dynamic_slice_in_dim(log_ps, start, tau, axis=0)
                )

            return vmap(sample_window)(starts)

        batch = vmap(sample_buffer)(keys, observations, actions, rewards, discounts, log_probs)
        batch = tuple(x.reshape((N * B,) + x.shape[2:]) for x in batch)

        observations, actions, rewards, discounts, log_probs = batch
        observations = self.env.preprocess_observation(observations)

        return (observations, actions, rewards, discounts, log_probs)

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
        obs, act, rew, flags, log_ps = replay_buffer["data"]

        new_obs = new_obs[:, None]
        new_act = new_act[:, None]
        new_rew = new_rew[:, None]
        new_flags = new_flags[:, None]
        new_log_ps = new_log_ps[:, None]

        def append_to_available_space(_):
            updated_obs = jax.lax.dynamic_update_slice_in_dim(obs, new_obs, size + 1, axis=1)
            updated_act = jax.lax.dynamic_update_slice_in_dim(act, new_act, size, axis=1)
            updated_rew = jax.lax.dynamic_update_slice_in_dim(rew, new_rew, size, axis=1)
            updated_flags = jax.lax.dynamic_update_slice_in_dim(flags, new_flags, size, axis=1)
            updated_log_ps = jax.lax.dynamic_update_slice_in_dim(log_ps, new_log_ps, size, axis=1)
            return updated_obs, updated_act, updated_rew, updated_flags, updated_log_ps, size + 1

        def append_at_capacity(_):
            return (
                jnp.concatenate((obs[:, 1:], new_obs), axis=1),
                jnp.concatenate((act[:, 1:], new_act), axis=1),
                jnp.concatenate((rew[:, 1:], new_rew), axis=1),
                jnp.concatenate((flags[:, 1:], new_flags), axis=1),
                jnp.concatenate((log_ps[:, 1:], new_log_ps), axis=1),
                jnp.asarray(C, dtype=jnp.int32),
            )

        obs, act, rew, flags, log_ps, size = jax.lax.cond(
            size < C,
            append_to_available_space,
            append_at_capacity,
            operand=None,
        )
        return {"data": (obs, act, rew, flags, log_ps), "size": size}


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
        return new_params, new_opt_states, loss, aux["filtered_means"]


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
