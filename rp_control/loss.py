import jax 
import jax.numpy as jnp
import jax.random as jr

from jax.random import PRNGKey

from rp_ssm.recognition.rpm import RPSSM

from rp_control.actor import Actor
from rp_control.critic import Critic
from rp_control.environment import Environment


class ActorCriticLoss:

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

    def init(self, key, observation, rpm_params):
        
        actor_init_key, critic_init_key = jr.split(key)

        posterior = self.rpm.initial_distribution(rpm_params, observation)
        mean = posterior.params["mean"]

        _actor_params = self.actor.init(actor_init_key, mean)
        _critic_params = self.critic.init(critic_init_key, mean)

        return {"actor": _actor_params, "critic": _critic_params, "rpm": rpm_params}

    def grads(self, key, params, posterior, env_state):

        # calculate d log[p(a | s)]
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
        return self.rpm.filter(params, state, observation)
    
    def episode(self, key, params):
        """
        Generate next sample from the environment using Actor and environment
        """

        def _episode(carry, inp):
            _mkey, _okey = inp
            _post, _env = carry

            # actor takes action
            mean = _post.params["mean"]
            _, action = self.actor.apply(params["actor"], mean)

            # action influences environment. Observe resultant state
            _env_p1, _ = self.env.model(_mkey, _env, action)
            _obs_p1 = self.env.observe(_okey, _env_p1)

            # use RPM to get posterior latents under observation
            _post_p1 = self.get_posterior(params["rpm"], _post, _obs_p1)

            return (_post_p1, _env_p1), (_env_p1, _obs_p1, action)

        model_key, obs_key, init_key = jr.split(key, 3)
        model_keys = jr.split(model_key, self.env.T - 1)
        obs_keys = jr.split(obs_key, self.env.T)

        # carry 0 stuff
        initial_state = self.env.initial_state(init_key)
        initial_obs = self.env.observe(obs_keys[0], initial_state)
        initial_posterior = self.get_initial_distribution(params, initial_obs)

        # scan for episode
        carry_0 = (initial_posterior, initial_state)
        inps = (model_keys, obs_keys[1:])
        _, (env_states, observations, actions) = jax.lax.scan(_episode, carry_0, inps)

        # insert t=0 
        env_states = jnp.concatenate([initial_state[None], env_states], axis=0)
        observations = jnp.concatenate([initial_obs[None], observations], axis=0)
        return env_states, observations, actions