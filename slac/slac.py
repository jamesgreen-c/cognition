import jax
import jax.numpy as jnp
import jax.random as jr

from jax import Array
from jax.random import PRNGKey
from jax.lax import stop_gradient

from jax.tree_util import tree_map

from rp_ssm.recognition.rpm import RPSSM

from slac.actor import Actor
from slac.critic import Critic
from slac.environment import Environment


class SLAC:

    def __init__(
            self, 
            actor: Actor, 
            critic: Critic, 
            rpm: RPSSM,
            environment: Environment
        ):

        self.actor = actor
        self.critic = critic
        self.rpm = rpm
        self.env = environment

    def init(self, state, rpm_params):
        key, actor_init_key, critic_init_key = jr.split(jr.PRNGKey(self.config.seed), 3)
        _actor_params = self.actor.init(actor_init_key, state)
        _critic_params = self.critic.init(critic_init_key, state)

        self.params = {"actor": _actor_params, "critic": _critic_params, "rpm": rpm_params}

    def grads(self, key, params, posterior, env_state):

        # sample action and calculate d log[p(a | s)]
        mean = posterior.params["mean"]
        (log_prob, action), actor_grad = jax.value_and_grad(self.actor.apply, argnums=0, has_aux=True)(params["actor"], mean) 

        # observe next_state and reward
        model_key, obs_key = jr.split(key)
        next_env_state, reward = self.env.model(model_key, env_state, action)

        # apply RPM to observation to get next state representation
        observation = self.env.observe(obs_key, next_env_state)
        next_posterior = self.get_posterior(params["rpm"], posterior, observation)
        next_mean = next_posterior.params["mean"]

        # calculate estimated values for S and S'
        value, critic_grad = jax.value_and_grad(self.critic.apply, argnums=0)(params["critic"], mean) 
        next_value = self.critic.apply(params["critic"], next_mean)

        grads = {"actor": actor_grad, "critic": critic_grad}
        return grads, (next_posterior, next_env_state, reward, next_value, value)

    def get_initial_distribution(self, params, observation):
        return self.rpm.initial_distribution(params["rpm"], observation)
    
    def get_posterior(self, params, state, observation):
        return self.rpm.filter(params["rpm"], state, observation)
    
    def apply(self):
        """
        Generate samples from the environment using Actor and environment
        """
        pass