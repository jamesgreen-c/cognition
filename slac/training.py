"""
Trainer job:

1. Needs access to data generation process given an action and current state
2. Needs access to reward sampling process given resultant state



3. define a jitted training step that:
    - applies the RPM to observations to get latents
    - applies actor to latents to generate next state
    - applies critic to next states to get values estimates
    - samples true reward of next state
    - applies semi-gradient updates to both the actor and critic networks
    - returns (action, next_state, reward) for tracking and next training step

4. needs a config for:
    - alphas (step sizes for semi-gradient steps) not the same as learning rates
    - batch size 
    - debug
    - optimiser: adam 
    - num-iter: int = 1? 
"""

import optax

import jax
import jax.random as jr
import jax.numpy as jnp

from tqdm import tqdm
from dataclasses import dataclass, field

from jax import Array
from jax.lax import stop_gradient as stop_grad
from jax.tree_util import tree_map

from slac.slac import SLAC
from slac.config import Config
from slac.environment import Environment


class Trainer:

    opt_states: dict[str, optax.OptState]
    itr: int

    def __init__(
        self, 
        slac: SLAC, 
        config: Config, 
        environment: Environment
    ):

        self.config = config
        self.slac = slac

        self.actor_opt = config.actor.build()
        self.critic_opt = config.critic.build()

        self.env = environment

        self.itr = 0

    def train_step(
            self, 
            key, 
            params: dict, 
            opt_states: dict, 
            traces: dict,
            reward_average: Array,
            posterior: Array,
            env_state: Array
        ):
        """
        
        data:  (D) previous states
        """

        # apply actor and critic
        grads, aux = self.slac.grads(key, params, posterior, env_state)
        next_posterior, next_env_state, reward, next_value, value = aux

        # continuous eligibility trace updates
        delta = stop_grad(reward - reward_average + next_value - value)   # calculate delta using TD(1)
        traces = self._update_traces(traces, grads)                       # update eligiblity traces
        actor_loss = tree_map(lambda z: -delta * z, traces["actor"])
        critic_loss = tree_map(lambda z: -delta * z, traces["critic"])

        # apply semi-gradient updates
        new_params, new_opt_states = self._update_params(actor_loss=actor_loss, 
                                                         critic_loss=critic_loss,
                                                         params=params,
                                                         opt_states=opt_states)
        
        return new_params, new_opt_states, next_posterior, next_env_state, delta, traces


    def fit(self, rpm_params, env_state, initial_data):
        """
        Parameters
        ----------
        initial_env_state:  (*K)  The true initial environment state
        initial_data:       (*D)  T=0 observation of the environment
        """

        train_step = jax.jit(self.train_step) if not self.config.debug else self.train_step

        # get initial posterior latent
        posterior = self.slac.get_initial_distribution({"rpm": rpm_params}, initial_data)   # or something along these lines (will break currently)
        mean = posterior.params["mean"]

        # initialise actor and critic networks
        key, init_key = jr.split(jr.PRNGKey(self.config.seed), 2)
        self.params = self.slac.init(init_key, mean, rpm_params)
        self.opt_states = {"actor": self.actor_opt.init(self.params["actor"]),
                           "critic": self.critic_opt.init(self.params["critic"])}

        # stores
        reward_average = 0
        traces = {"actor": tree_map(jnp.zeros_like, self.params["actor"]),
                  "critic": tree_map(jnp.zeros_like, self.params["critic"])}
        
        # run
        pbar = tqdm(range(self.config.num_iter))
        for self.itr in pbar:
            key, subkey = jr.split(key)
 
            self.params, self.opt_states, posterior, env_state, delta, traces = train_step(
                subkey,
                self.params,
                self.opt_states,
                traces,
                reward_average,
                posterior,
                env_state
            )

            # update reward average
            reward_average = reward_average + self.config.reward_lr * delta

        return self.params

    def _update_trace(self, decay_rate, trace, gradient):
        return tree_map(lambda z, g: decay_rate * z + g, trace, gradient)

    def _update_traces(self, traces, grads):
        trace_actor = self._update_trace(self.config.actor_trace_decay, traces["actor"], grads["actor"])
        trace_critic = self._update_trace(self.config.critic_trace_decay, traces["critic"], grads["critic"])
        return {"actor": trace_actor, "critic": trace_critic}

    def _update_params(self, actor_loss, critic_loss, params, opt_states):

        actor_update, actor_opt_state = self.actor_opt.update(actor_loss, opt_states["actor"], params["actor"])
        critic_update, critic_opt_state = self.critic_opt.update(critic_loss, opt_states["critic"], params["critic"])

        actor_params = optax.apply_updates(params["actor"], actor_update)
        critic_params = optax.apply_updates(params["critic"], critic_update)

        new_opt_states = {"actor": actor_opt_state, "critic": critic_opt_state}
        new_params = {"actor": actor_params, "critic": critic_params}

        return new_params, new_opt_states