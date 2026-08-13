import optax

import jax
import jax.random as jr
import jax.numpy as jnp

from tqdm import tqdm
from typing import Callable, Any

from jax import Array, vmap
from jax.random import PRNGKey
from jax.tree_util import tree_map
from jax.lax import stop_gradient as stopgrad

from rp_slac.environment import Environment
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
            environment: Environment,
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


    def experience_step(self, key: PRNGKey, params: dict, env_states: Array, data: tuple[Array]):

        K = self.config.collection_steps
        policy = lambda _k, _state: self.control.policy(_k, params, _state)

        # sample K new steps from environment
        if self.config.actor_state == "observation":
                
            M = env_states.shape[0]
            env_states, _, observations, actions, rewards, flags = self.env.sample(
                key=key, 
                policy=policy, 
                num_samples=M, 
                num_steps=K, 
                state_0=env_states,
                actor_state=self.config.actor_state
            )

        else:
            # TODO: if actor state is the RPM latent, need to do a new method that has access to the model during env sampling
            raise NotImplementedError
        
        # rolling window
        observations = jnp.concatenate((data[0][:, K:], observations[:, 1:]), axis=1)
        actions = jnp.concatenate((data[1][:, K:], actions), axis=1)
        rewards = jnp.concatenate((data[2][:, K:], rewards), axis=1)
        flags = jnp.concatenate((data[3][:, K:], flags), axis=1)
        replay_buffer = (observations, actions, rewards, flags)

        return replay_buffer, env_states
    
    def pretrain_step(self, itr: int, params: AllParams, opt_states: dict[optax.OptState], data: tuple[Array]):
        new_params, new_opt_states, model_loss, _ = self._update_model(itr, params, opt_states, data)
        return new_params, new_opt_states, model_loss

    def train_step(
            self,
            key: PRNGKey,
            itr: int,
            params: AllParams,
            opt_states: dict[optax.OptState],
            data: tuple[Array]
    ) -> tuple[dict, dict[optax.OptState], dict[float]]: 
        sample_key, actor_key, critic_key = jr.split(key, 3)

        # run model updates and sample latents
        params, opt_states, model_loss, posterior = self._update_model(itr, params, opt_states, data)
        latents = posterior.sample(sample_key)

        # run soft actor-critic updates
        params, opt_states, critic_loss = self._update_critic(critic_key, params, opt_states, latents, data)
        params, opt_states, actor_loss = self._update_actor(actor_key, params, opt_states, latents, data)

        losses = {"model": model_loss, "critic": critic_loss, "actor": actor_loss}
        return params, opt_states, losses
    

    def fit(self, use_pbar: bool = True) -> None:
        """ 
        
        Parameters 
        ----------
        replay_buffer:  Pregenerated data from a random policy 
        """
        key, buffer_key, init_key = jr.split(jr.PRNGKey(self.config.seed), 3)

        replay_buffer, env_states = self._init_replay_buffer(buffer_key)
        self.params, self.opt_states, self.opts = self.init(init_key, replay_buffer)

        experience_step = jax.jit(self.experience_step) if self.config.jit else self.experience_step
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
        self.average_rewards = []
        
        pbar = tqdm(range(self.config.num_iter), disable=not(use_pbar), desc="Training")
        for self.itr in pbar:
            key, collection_key, batch_key, train_key = jr.split(key, 4)

            replay_buffer, env_states = experience_step(collection_key, self.params, env_states, replay_buffer)
            batch = self._get_batch(batch_key, replay_buffer)
            
            self.params, self.opt_states, losses = train_step(train_key, self.itr, self.params, self.opt_states, batch)
            self._stabilise_params()

            # calculate average reward
            average_reward = replay_buffer[2].mean()

            self.model_losses.append(losses["model"])
            self.critic_losses.append(losses["critic"])
            self.actor_losses.append(losses["actor"])
            self.average_rewards.append(self.average_rewards)

            pbar.set_postfix(
                average_reward=average_reward, 
                model_loss=losses["model"], 
                critic_loss=losses["critic"], 
                actor_loss=losses["actor"]
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
        """ Initialise the replay buffer to capacity """
        env_states, _, observations, actions, rewards, flags = self.env.sample(
            key=key,
            policy=self.env.random_action,
            num_samples=self.config.num_buffers,
            num_steps=self.config.capacity,
        )
        return (observations, actions, rewards, flags), env_states

    def _get_batch(self, key: PRNGKey, replay_buffer: tuple[Array]):
        """
        Sample B contiguous windows independently from each of N buffers.

        Returns
        -------
        observations: (N * B, tau + 1, ...)
        actions:      (N * B, tau, ...)
        rewards:      (N * B, tau)
        discounts:    (N * B, tau)
        """
        observations, actions, rewards, discounts = replay_buffer

        N = self.config.num_buffers
        B = self.config.batch_size
        tau = self.config.sequence_length
        capacity = actions.shape[1]

        if observations.shape[:2] != (N, capacity + 1):
            raise ValueError("Expected observations shaped (N, C + 1, ...), received {}.".format(observations.shape))

        num_starts = capacity - tau + 1
        if num_starts <= 0:
            raise ValueError("Replay buffers are shorter than the sampled window.")

        keys = jr.split(key, N)

        def sample_buffer(key, obs, act, rew, disc):
            starts = jr.randint(key, (B,), 0, num_starts)

            def sample_window(start):
                return (
                    jax.lax.dynamic_slice_in_dim(obs, start, tau + 1, axis=0),
                    jax.lax.dynamic_slice_in_dim(act, start, tau, axis=0),
                    jax.lax.dynamic_slice_in_dim(rew, start, tau, axis=0),
                    jax.lax.dynamic_slice_in_dim(disc, start, tau, axis=0),
                )

            return vmap(sample_window)(starts)

        batch = vmap(sample_buffer)(keys, observations, actions, rewards, discounts)
        return tuple(x.reshape((N * B,) + x.shape[2:]) for x in batch)


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

        # return updated params and opt_states
        new_params = {**params, **new_params}
        new_opt_states = {**opt_states, **new_opt_states}
        return new_params, new_opt_states, loss, aux["posterior"]


    def _update_critic(self, key, params, opt_states, latents, data):
        """ Update soft critic parameters """
        rho = self.config.target_update_rate
        _params = {"actor": params["actor"], "critic": params["critic"]}

        # latest param updates
        loss, grads = jax.value_and_grad(self.control.critic_loss, argnums=1)(key, _params, latents, data)
        latest_updates, latest_opt_states = self.opts["critic"].update(grads["critic"]["latest"], 
                                                                       opt_states["critic"]["latest"], 
                                                                       params["critic"]["latest"])
        latest_params = optax.apply_updates(params["critic"]["latest"], latest_updates)

        # target param updates
        target_update = lambda _tps, _lps: (1.0 - rho) * _tps + rho * _lps
        target_params = tree_map(target_update, params["critic"]["target"], latest_params)

        # return updated params and opt_states
        critic_params = {"latest": latest_params, "target": target_params}
        critic_opt_states = {"latest": latest_opt_states, "target": opt_states["critic"]["target"]}
        new_params = {**params, "critic": critic_params}
        new_opt_states = {**opt_states, "critic": critic_opt_states}
        return new_params, new_opt_states, loss


    def _update_actor(self, key, params, opt_states, latents, data):
        """ Update soft actor parameters """

        _params = {"actor": params["actor"], "critic": params["critic"]}
        loss, grads = jax.value_and_grad(self.control.actor_loss, argnums=1)(key, _params, latents, data)
        updates, new_actor_opt_states = self.opts["actor"].update(grads["actor"], opt_states["actor"], params["actor"])
        actor_params = optax.apply_updates(params["actor"], updates)

        new_params = {**params, "actor": actor_params}
        new_opt_states = {**opt_states, "actor": new_actor_opt_states}
        return new_params, new_opt_states, loss


    def _stabilise_params(self):
        # TODO stabilisation would require new Q calculation so needs to go into prior? 
        if self.config.stabilise_A == 'scale':
            self.params["prior"]["A"] = scale_sv(self.params["prior"]["A"], EPS)
        elif self.config.stabilise_A == 'clip':
            self.params["prior"]["A"] = clip_sv(self.params["prior"]["A"], EPS)


    def apply(self, data: tuple[Array]) -> None:
        pass



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