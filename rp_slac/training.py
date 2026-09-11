import optax

import jax
import jax.random as jr
import jax.numpy as jnp

from tqdm import tqdm
from typing import Callable, Union

from jax import Array, vmap
from jax.random import PRNGKey
from jax.tree_util import tree_map
from jax.lax import stop_gradient as stopgrad

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
            environment: Union[JAXEnvironment, MuJoCoSimulationEnvironment],
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

        if self.config.initial_steps < self.config.sequence_length:
            raise ValueError("initial_steps must be at least sequence_length.")

        if self.config.initial_steps > self.config.capacity:
            raise ValueError("initial_steps cannot exceed replay capacity.")

        if self.config.collection_steps > self.config.capacity:
            raise ValueError("collection_steps cannot exceed replay capacity.")


    def experience_step(
            self,
            key: PRNGKey,
            params: dict,
            env_states: Array,
            replay_buffer: dict,
        ):
        K = self.config.collection_steps
        C = self.config.capacity
        M = self.config.num_buffers
        size = replay_buffer["size"]

        def policy(_key, _observation):
            _observation = self.env.preprocess_observation(_observation)
            return self.control.policy(_key, params, _observation)

        env_states, _, new_obs, new_act, new_rew, new_flags, new_log_ps = self.env.sample(
            key=key,
            policy=policy,
            num_samples=M,
            num_steps=K,
            state_0=env_states,
        )
        
        obs, act, rew, flags, log_ps = replay_buffer["data"]

        def append_to_available_space(_):
            updated_obs = jax.lax.dynamic_update_slice_in_dim(obs, new_obs[:, 1:], size + 1, axis=1)
            updated_act = jax.lax.dynamic_update_slice_in_dim(act, new_act, size, axis=1)
            updated_rew = jax.lax.dynamic_update_slice_in_dim(rew, new_rew, size, axis=1)
            updated_flags = jax.lax.dynamic_update_slice_in_dim(flags, new_flags, size, axis=1)
            updated_log_ps = jax.lax.dynamic_update_slice_in_dim(log_ps, new_log_ps, size, axis=1)
            return (updated_obs, updated_act, updated_rew, updated_flags, updated_log_ps, size + K)

        def append_at_capacity(_):
            keep = C - K
            start = size - keep

            kept_obs = jax.lax.dynamic_slice_in_dim(obs, start, keep + 1, axis=1)
            kept_act = jax.lax.dynamic_slice_in_dim(act, start, keep, axis=1)
            kept_rew = jax.lax.dynamic_slice_in_dim(rew, start, keep, axis=1)
            kept_flags = jax.lax.dynamic_slice_in_dim(flags, start, keep, axis=1)
            kept_log_ps = jax.lax.dynamic_slice_in_dim(log_ps, start, keep, axis=1)

            return (
                jnp.concatenate((kept_obs, new_obs[:, 1:]), axis=1),
                jnp.concatenate((kept_act, new_act), axis=1),
                jnp.concatenate((kept_rew, new_rew), axis=1),
                jnp.concatenate((kept_flags, new_flags), axis=1),
                jnp.concatenate((kept_log_ps, new_log_ps), axis=1),
                jnp.asarray(C, dtype=jnp.int32),
            )

        obs, act, rew, flags, log_ps, size = jax.lax.cond(
            size + K <= C,
            append_to_available_space,
            append_at_capacity,
            operand=None,
        )

        replay_buffer = {"data": (obs, act, rew, flags, log_ps), "size": size}
        return replay_buffer, env_states
    

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
        sample_key, actor_key, critic_key = jr.split(key, 3)

        # calculate all updates from the previous step parameters
        model_params, model_opt_states, model_loss, posterior = self._update_model(itr, params, opt_states, data)
        latents = posterior.sample(sample_key)    # latents = posterior.params["means"]

        critic_params, critic_opt_state, critic_loss = self._update_critic(critic_key, params, opt_states, latents, data)
        actor_params, actor_opt_state, actor_loss, aux = self._update_actor(actor_key, params, opt_states, latents, data)
        log_alpha, alpha_opt_state, alpha_loss = self._update_alpha(params, opt_states, data)

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

        losses = {"model": model_loss, "critic": critic_loss, "actor": actor_loss, "alpha": alpha_loss}
        return new_params, new_opt_states, losses, aux
    

    def fit(self, use_pbar: bool = True) -> None:
        """ 
        
        Parameters 
        ----------
        replay_buffer:  Pregenerated data from a random policy 
        """
        key, buffer_key, init_key = jr.split(jr.PRNGKey(self.config.seed), 3)

        replay_buffer, env_states = self._init_replay_buffer(buffer_key)
        self.params, self.opt_states, self.opts = self.init(init_key, replay_buffer["data"])

        # experience_step = jax.jit(self.experience_step) if self.config.jit else self.experience_step
        experience_step = (
            jax.jit(self.experience_step)
            if self.config.jit and self.env.jax_compatible
            else self.experience_step
        )
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
            key, collection_key, batch_key, train_key = jr.split(key, 4)

            replay_buffer, env_states = experience_step(collection_key, self.params, env_states, replay_buffer)
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
    
    def _init_replay_buffer(self, key: PRNGKey):
        N = self.config.num_buffers
        I = self.config.initial_steps
        C = self.config.capacity

        env_states, _, observations, actions, rewards, flags, log_probs = self.env.sample(
            key=key,
            policy=self.env.random_action,
            num_samples=N,
            num_steps=I,
        )

        # initialise buffers with full capacity
        observation_buffer = jnp.zeros((N, C + 1) + observations.shape[2:], dtype=observations.dtype)
        action_buffer = jnp.zeros((N, C) + actions.shape[2:], dtype=actions.dtype)
        reward_buffer = jnp.zeros((N, C), dtype=rewards.dtype)
        flag_buffer = jnp.zeros((N, C), dtype=flags.dtype)
        log_prob_buffer = jnp.zeros((N, C), dtype=log_probs.dtype)

        # store initial experience
        observation_buffer = observation_buffer.at[:, :I + 1].set(observations)
        action_buffer = action_buffer.at[:, :I].set(actions)
        reward_buffer = reward_buffer.at[:, :I].set(rewards)
        flag_buffer = flag_buffer.at[:, :I].set(flags)
        log_prob_buffer = log_prob_buffer.at[:, :I].set(log_probs)

        replay_buffer = {
            "data": (
                observation_buffer,
                action_buffer,
                reward_buffer,
                flag_buffer,
                log_prob_buffer,
            ),
            "size": jnp.asarray(I, dtype=jnp.int32),
        }
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
            updates, new_opt_states[name] = self.opts[name].update(
                grads[name],
                opt_states[name],
                params[name]
            )
            new_params[name] = optax.apply_updates(params[name], updates)

        return new_params, new_opt_states, loss, aux["posterior"]


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


    def _update_alpha(self, params, opt_states, data: tuple[Array]):
        """ Update log temperature parameter """
        _params = {"log_alpha": params["log_alpha"]}
        loss, grads = jax.value_and_grad(self.control.alpha_loss)(_params, data)

        updates, alpha_opt_state = self.opts["alpha"].update(grads["log_alpha"], 
                                                             opt_states["alpha"], 
                                                             params["log_alpha"])

        log_alpha = optax.apply_updates(params["log_alpha"], updates)
        return log_alpha, alpha_opt_state, loss
    

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



########### OLD SEQUENTIAL UPDATES #############

    # def pretrain_step(self, itr: int, params: AllParams, opt_states: dict[optax.OptState], data: tuple[Array]):
    #     new_params, new_opt_states, model_loss, _ = self._update_model(itr, params, opt_states, data)
    #     return new_params, new_opt_states, model_loss


    # def train_step(
    #         self,
    #         key: PRNGKey,
    #         itr: int,
    #         params: AllParams,
    #         opt_states: dict[optax.OptState],
    #         data: tuple[Array]
    # ) -> tuple[dict, dict[optax.OptState], dict[float]]: 
    #     sample_key, actor_key, critic_key = jr.split(key, 3)

    #     # run model updates and sample latents
    #     params, opt_states, model_loss, posterior = self._update_model(itr, params, opt_states, data)
    #     # latents = posterior.params["means"]
    #     latents = posterior.sample(sample_key)

    #     # run soft actor-critic updates
    #     params, opt_states, critic_loss = self._update_critic(critic_key, params, opt_states, latents, data)
    #     params, opt_states, actor_loss, aux = self._update_actor(actor_key, params, opt_states, latents, data)
    #     params, opt_states, alpha_loss = self._update_alpha(params, opt_states, data)

    #     losses = {"model": model_loss, "critic": critic_loss, "actor": actor_loss, "alpha": alpha_loss}
    #     return params, opt_states, losses, aux



    # def _update_model(self, itr: int, params, opt_states, data):
    #     """ Update RPM model parameters using Kalman smoothing """

    #     beta = self.config.beta_schedule(itr)
    #     em = self.config.em

    #     obs = data[0]
    #     actions = data[1]

    #     _params = {"prior": params["prior"], "rpm": params["rpm"]}
    #     (loss, aux), grads = jax.value_and_grad(self.model.loss, has_aux=True)(_params, obs, actions, beta, em)

    #     new_params = {}
    #     new_opt_states = {}
    #     for name in ("prior", "rpm"):
    #         updates, new_opt_states[name] = self.opts[name].update(
    #             grads[name],
    #             opt_states[name],
    #             params[name]
    #         )
    #         new_params[name] = optax.apply_updates(params[name], updates)

    #     # return updated params and opt_states
    #     new_params = {**params, **new_params}
    #     new_opt_states = {**opt_states, **new_opt_states}
    #     return new_params, new_opt_states, loss, aux["posterior"]


    # def _update_critic(self, key, params, opt_states, latents, data):
    #     """ Update soft critic parameters """
    #     rho = self.config.target_update_rate
    #     _params = {"actor": params["actor"], "critic": params["critic"], "log_alpha": params["log_alpha"]}

    #     # latest param updates
    #     loss, grads = jax.value_and_grad(self.control.critic_loss, argnums=1)(key, _params, latents, data)
    #     latest_updates, new_critic_opt_states = self.opts["critic"].update(grads["critic"]["latest"], 
    #                                                                        opt_states["critic"], 
    #                                                                        params["critic"]["latest"])
    #     latest_params = optax.apply_updates(params["critic"]["latest"], latest_updates)

    #     # target param updates
    #     target_update = lambda _tps, _lps: (1.0 - rho) * _tps + rho * _lps
    #     target_params = tree_map(target_update, params["critic"]["target"], latest_params)

    #     # return updated params and opt_states
    #     critic_params = {"latest": latest_params, "target": target_params}
    #     new_params = {**params, "critic": critic_params}
    #     new_opt_states = {**opt_states, "critic": new_critic_opt_states}
    #     return new_params, new_opt_states, loss


    # def _update_actor(self, key, params, opt_states, latents, data):
    #     """ Update soft actor parameters """

    #     _params = {"actor": params["actor"], "critic": params["critic"], "log_alpha": params["log_alpha"]}
    #     (loss, aux), grads = jax.value_and_grad(
    #         self.control.actor_loss, argnums=1, has_aux=True
    #     )(key, _params, latents, data)

    #     updates, new_actor_opt_states = self.opts["actor"].update(grads["actor"], opt_states["actor"], params["actor"])
    #     actor_params = optax.apply_updates(params["actor"], updates)

    #     new_params = {**params, "actor": actor_params}
    #     new_opt_states = {**opt_states, "actor": new_actor_opt_states}
    #     return new_params, new_opt_states, loss, aux


    # def _update_alpha(self, params, opt_states, data: tuple[Array]):
    #     """ Update log temperature parameter """
    #     _params = {"log_alpha": params["log_alpha"]}
    #     loss, grads = jax.value_and_grad(self.control.alpha_loss)(_params, data)

    #     updates, alpha_opt_state = self.opts["alpha"].update(grads["log_alpha"], 
    #                                                          opt_states["alpha"], 
    #                                                          params["log_alpha"])

    #     log_alpha = optax.apply_updates(params["log_alpha"], updates)
    #     new_params = {**params,"log_alpha": log_alpha}
    #     new_opt_states = {**opt_states, "alpha": alpha_opt_state}
    #     return new_params, new_opt_states, loss



 # def train_continue(self, data: tuple[Array], new_iter: int, key: Array, y: Array = None):
    #     train_step = jax.jit(self.train_step) if self.config.jit else self.train_step

    #     pbar = tqdm(range(self.itr, self.itr + new_iter))
    #     for self.itr in pbar:
    #         key, subkey = jr.split(key)
    #         batch_indices = jr.randint(subkey, (self.config.batch_size,), 0, data[0].shape[0])
    #         data_batch = [d[batch_indices] for d in data]

    #         loss, aux, self.params, self.opt_states = train_step(
    #             self.params, self.opt_states, data_batch
    #         )
            
    #         self._stabilise_params()

    #         self.loss_tot.append(loss)
    #         to_print = self.logger(self, aux, batch_indices) # TODO: validation step?
    #         to_print.update({'loss': f'{loss:.3f}'})

    #         pbar.set_postfix(**to_print)

    #         if y is not None and self.itr % 100 == 0:
    #             x = self.apply((data[0], ))[1].params["means"]
    #             r2 = linear_r2(x, y)
    #             self.r2_history.append(r2)



        # if K > 1:

        #     def _body():
        #         pass
        #     # only lax scan if more than one step 

        # # otherwise just do one update


        # # update target as
        # target_params_1 = self.params["critic"]["target"]["one"]
        # target_params_2 = self.params["critic"]["target"]["two"]

        # latest_params_1 = self.params["latest"]["target"]["one"]
        # latest_params_2 = self.params["latest"]["target"]["two"]

        # tar_params_1 = (1 - rho) * tar_params_1 + rho * latest_params_1
        # tar_params_2 = (1 - rho) * tar_params_2 + rho * latest_params_2

        # params["critic"]["target"] = {"one": tar_params_1, "two": tar_params_2}
        # return params