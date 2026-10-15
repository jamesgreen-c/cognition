"""RAMP interior free energy with an optional causal reward-prediction loss.

The empirical RAMP objective is unchanged. Its Gaussian calculations use
centred means and Cholesky solves, and its pairwise terms are rematerialised
one time/branch/query at a time. No images are reconstructed.

Reward supervision is an auxiliary likelihood on filtered latent pairs and
recorded actions. Rewards never enter inference or deployed policy inputs.
"""

from typing import Any

import jax
import jax.numpy as jnp

from jax import Array, lax, vmap
from jax.random import PRNGKey
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

from rp_slac.config import Config
from rp_slac.distributions import AllParam
from rp_slac.recognition.ramp import RAMPSSM


class ConstrainedIVFreeEnergy:

    def __init__(self, model: RAMPSSM, reward_weight: float = 1.0):
        if reward_weight < 0:
            raise ValueError('reward_weight must be nonnegative.')
    
        self.model = model
        self.reward_weight = reward_weight

    def configure(self, config: Config):
        # self.num_timesteps = config.sequence_length + 1
        self.batch_size = config.model_batch_size
        # self.num_buffers = config.num_buffers
        self.num_factors = len(self.model.recognition) + 2

    def init(self, key: Array, data: tuple[Array, ...], config: Config):
        self.configure(config)
        params = self.model.init(key, data[0], config)
        opts = {'prior': config.prior.build(), 'rpm': config.recognition.build()}
        opt_states = {name: opts[name].init(params[name]) for name in opts}
        return params, opt_states, opts

    def get_posterior(self, prior_params, rec_params, actions, observations):
        """
        Form smoothed and causal Gaussian node marginals from message sums.

        q_t = f0 + observations_t + left_t + right_t in natural parameters.
        The filtered belief excludes right_t, so it cannot use future data.
        The posterior has mean/covariance, without cross-time covariance blocks.

        Parameters
        ----------
        prior_params:  Model params: 'forward' and 'backward' message-network parameters.
        rec_params:    sequence of length J. Observation-network Flax variable trees.
        actions:       (B,T-1,A), (B,T-1) when A=1, or None when A=0. Recorded edge actions.
        observations:  (B,T,*O_1) or sequence of J arrays (B,T,*O_j). 
                            Replay observations with common batch and time axes.

        Returns
        -------
        output: tuple (prior, messages, posterior, filtered)
            prior:     GaussianNatParam,  pwm=(N,),      p=(N,N), shared across nodes.
            messages:  GaussianNatParam,  pwm=(K,B,T,N), p=(K,B,T,N,N).
            posterior: GaussianDistParam, mean=(B,T,N),  cov=(B,T,N,N).
            filtered:  GaussianDistParam, mean=(B,T,N),  cov=(B,T,N,N).
            features:  Array (B,T,N) of the feature vectors produced dense layer of the REC RPM.
        """
        # apply recognition to data to get factors over latent nodes. Apply BP to find messages.
        factors_nat, features = self.model.get_factors(rec_params, observations, return_features=True)
        messages = self.model.get_messages(prior_params, factors_nat, actions)

        # calculate posterior 
        prior = self.model.prior
        posterior = (prior + messages.sum(axis=0)).dist_param

        # additionally calculate causal filter distribution
        left = tree_map(lambda x: x[-2], messages)
        filtered = (prior + factors_nat.sum(axis=0) + left).dist_param

        return prior, messages, posterior, filtered, features

    def loss(
        self, params: dict, observations: Array, actions: Array | None,
        beta: float = 1.0, em: bool = False, rewards: Array | None = None,
        discounts: Array | None = None, key: Array | None = None,
    ) -> tuple[Array, Any]:
        
        prior, messages, posterior, filtered, features = self.get_posterior(
            params['prior'], params['rpm'], actions, observations
        )

        # In EM mode hold the E-step belief fixed, but still train the messages.
        posterior = lax.cond(em, lax.stop_gradient, lambda q: q, posterior)
        filtered = lax.cond(em, lax.stop_gradient, lambda q: q, filtered)
        latents = filtered.sample(key, shape=None)

        # calculate RAMP loss and reward loss
        kl_qf, log_Gamma, kl_qp = self.get_model_loss_terms(prior, messages, posterior)
        reward_nll, reward_mse = self.get_reward_loss_terms(
            params, latents, key, actions, rewards, discounts,
        )

        batch, times = posterior.params['mean'].shape[:2]
        Z = batch * times  # One objective per node, not per incoming edge.
        ramp_loss = -(log_Gamma - kl_qf - beta * kl_qp) / Z
        loss = ramp_loss + self.reward_weight * reward_nll

        return loss, {
            'filter_posterior': filtered,
            'features': features,
            'ramp_loss': ramp_loss,
            'reward_nll': reward_nll,
            'reward_mse': reward_mse,
            'kl_qp': kl_qp / Z,
            'kl_qf': kl_qf / Z,
            'log_Gamma': log_Gamma / Z,
        }

    def get_model_loss_terms(self, prior, messages, posterior):
        """
        Return scalar sums (KL_qf, log_Gamma, KL_qp), as in model_fe.py.

        Parameters
        ----------
        prior:        GaussianNatParam. Fixed node prior with pwm=(N,), p=(N,N).
        factors_nat:  GaussianNatParam. All incoming ratios with pwm=(K,B,T,N), p=(K,B,T,N,N).
        posterior:    GaussianDistParam 
                        Node beliefs with mean=(B,T,N), cov=(B,T,N,N). B must be at least two.

        Returns
        -------
        output: tuple (kl_qf, log_Gamma, kl_qp)
            Three scalar arrays ().
            kl_qf and log_Gamma sum over branches, sequences and times.
            kl_qp sums over sequences and times only.
        """
        q = AllParam(posterior).nat_param
        if q.params['pwm'].shape[0] < 2:
            raise ValueError('RAMP learning needs at least two replay sequences.')
        messages_by_time = tree_map(lambda x: jnp.moveaxis(x, 2, 0), messages)
        q_by_time = tree_map(lambda x: jnp.swapaxes(x, 0, 1), q)

        # Map over time to avoid materialising all T batches of pairwise
        # Gaussian products. Rematerialise them during the backward pass.
        def node_terms(inputs):
            messages, node_q = inputs
            return self._node_loss_terms(prior, messages, node_q)

        terms = jax.vmap(node_terms)((messages_by_time, q_by_time))
        # terms = lax.map(jax.checkpoint(node_terms), (messages_by_time, q_by_time))
        return tuple(term.sum() for term in terms)

    def get_reward_loss_terms(
            self, params: dict, latents: Array, key = None, 
            actions: Array | None = None, rewards: Array | None = None, 
            discounts: Array | None = None,
        ):
        """
        Return reward negative log likelihood of the true reward at current predicted reward 
        distribution. Also calculate an MSE as the mean squared error of the true reward and 
        the mean predicted reward.

        Parameters
        ----------
        params:     Model parameters
        latents:    A sample of latents from the filter distribution given by the RAMPSSM
        key:        RNG
        actions:    (B,T-1,A), (B,T-1) when A=1, or None when A=0. Recorded edge actions.
        rewards:    (B,T-1) Array of rewards for given transitions.
        discounts:  (B,T-1) Array of discounts for given timesteps.

        Returns
        -------
        output:  tuple(rewards_nll, reward_mse)
        """

        reward_nll = jnp.asarray(0.0)
        reward_mse = jnp.asarray(0.0)
        batch, times = latents.shape[:2]

        if self.model.reward_network is not None:

            if rewards is None or actions is None or key is None:
                raise ValueError('Reward-supervised RAMP requires rewards, actions and a sampling key.')
            if rewards.shape != (batch, times - 1):
                raise ValueError('Expected rewards shaped (B,T-1).')
            if discounts is not None and discounts.shape != rewards.shape:
                raise ValueError('Reward and replay-flag shapes must match.')

            preds_nat = self.model.predict_rewards(params['prior'], latents, actions)
            preds = preds_nat.dist_param

            residual = rewards - preds.params["mean"][..., 0]
            nll = -vmap(vmap(lambda _pred, _reward: _pred.log_prob(_reward)))(preds, rewards[..., None])
            # nll = -preds.log_prob(rewards)
            # nll = 0.5 * (residual / std)**2 + jnp.log(std) + 0.5 * jnp.log(2.0 * jnp.pi)

            # Current replay flags identify reset edges, whose next frame is
            # a reset observation. The normal sampler excludes these edges.
            valid = jnp.ones_like(rewards) if discounts is None else (discounts > 0).astype(nll.dtype)
            denominator = jnp.maximum(valid.sum(), 1.0)
            reward_nll = (valid * nll).sum() / denominator
            reward_mse = (valid * residual**2).sum() / denominator

        return reward_nll, reward_mse

    @staticmethod
    def _node_loss_terms(prior, messages, q):
        """
        Evaluate all incoming branches at one time node using unbatched KL APIs.

        The batch indices n and m select sequences, not time nodes.
        Each auxiliary Gaussian q_n+r_jn is used only to evaluate the bound.

        Parameters
        ----------
        prior:     GaussianNatParam. Fixed node prior with pwm=(N,), p=(N,N).
        messages:  GaussianNatParam. Incoming ratios at this time, pwm=(K,B,N), p=(K,B,N,N).
        q:         GaussianNatParam. Node beliefs at this time, pwm=(B,N), p=(B,N,N).

        Returns
        -------
        output: tuple (kl_qf, log_Gamma, kl_qp)
            Three scalar arrays ().
            kl_qf and log_Gamma sum over K branches and B sequences.
            kl_qp sums over B sequences once.
        """
        kl_qp = vmap(lambda qn: qn.kl(prior))(q)
        batch_size = q.params['pwm'].shape[0]

        def factor_terms(ratios):
            kl_qf = vmap(lambda qn, rn: qn.kl(qn + rn))(q, ratios)
            phi_f = vmap(lambda rm: (prior + rm).lognormalizer)(ratios)
            # G[n,m]: query belief n, candidate subtree m, at this time/branch.
            G = vmap(
                lambda qn: vmap(lambda rm: (qn + rm).lognormalizer)(ratios)
            )(q) - phi_f[None]
            # Phi(q_n) cancels between the diagonal and its row normaliser.
            # +log(B) retains the empirical mixture's 1/B. model_fe.py omits
            # this constant; retaining it makes a zero boundary message neutral.
            log_Gamma = jnp.diag(G) - logsumexp(G, axis=1) + jnp.log(batch_size)
            return kl_qf.sum(), log_Gamma.sum()

        kl_qf, log_Gamma = vmap(factor_terms)(messages)
        return kl_qf.sum(), log_Gamma.sum(), kl_qp.sum()


    def apply(self, prior_params, rec_params, observations, actions):
        """
        Form causal Gaussian node marginals from message sums.
        For use in SAC training using only forward beliefs.
        
        q_t = f0 + observations_t + left_t + right_t in natural parameters.
        The filtered belief excludes right_t, so it cannot use future data.

        Parameters
        ----------
        prior_params:  Model params: 'forward' and 'backward' message-network parameters.
        rec_params:    sequence of length J. Observation-network Flax variable trees.
        actions:       (B,T-1,A), (B,T-1) when A=1, or None when A=0. Recorded edge actions.
        observations:  (B,T,*O_1) or sequence of J arrays (B,T,*O_j). 
                            Replay observations with common batch and time axes.

        Returns
        -------
        output: tuple (prior, filtered, features)
            prior:     GaussianNatParam,  pwm=(N,),      p=(N,N), shared across nodes.
            filtered:  GaussianDistParam, mean=(B,T,N),  cov=(B,T,N,N).
            features:  Array (B,T,N) of the feature vectors produced dense layer of the REC RPM.
        """

        # apply recognition to data to get factors over latent nodes. Get left to right causal messages
        factors_nat, features = self.model.get_factors(rec_params, observations, return_features=True)
        messages = self.model.get_causal_messages(prior_params, factors_nat, actions)

        # calculate causal filter distribution
        prior = self.model.prior
        filtered = (prior + messages.sum(axis=0)).dist_param

        return prior, filtered, features


    # @staticmethod
    # def _node_loss_terms(prior, messages, q):
    #     """Sequential branches/queries, with a vectorised candidate row.

    #     For s=q_n+r_m and f=f0+r_m, the centred score equals
    #     Phi(s)-Phi(f) minus a query-only constant. It therefore gives the
    #     same diagonal empirical-mixture objective without subtracting the
    #     large h.T @ inv(P) @ h normalisers.
    #     """
    #     kl_qp = vmap(lambda qn: qn.kl(prior))(q).sum()
    #     means_q, _ = vmap(precision_mean_and_chol)(q.params['p'], q.params['pwm'])
    #     count = q.params['pwm'].shape[0]
    #     indices = jnp.arange(count)

    #     def factor_terms(ratios):
    #         kl_qf = vmap(lambda qn, rn: qn.kl(qn + rn))(q, ratios).sum()
    #         factors = prior + ratios
    #         means_f, chols_f = vmap(precision_mean_and_chol)(
    #             factors.params['p'], factors.params['pwm']
    #         )
    #         logdets_f = 2.0 * jnp.log(jnp.diagonal(chols_f, axis1=-2, axis2=-1)).sum(axis=-1)

    #         def query_terms(inputs):
    #             index, qn, mean_q = inputs

    #             def score(rn, mean_f, chol_f, logdet_f):
    #                 summed = qn + rn
    #                 mean_s, chol_s = precision_mean_and_chol(
    #                     summed.params['p'], summed.params['pwm']
    #                 )
    #                 delta_s, delta_f = mean_q - mean_s, mean_q - mean_f
    #                 quadratic_s = jnp.sum((chol_s.T @ delta_s)**2)
    #                 quadratic_f = jnp.sum((chol_f.T @ delta_f)**2)
    #                 logdet_s = 2.0 * jnp.log(jnp.diag(chol_s)).sum()
    #                 return 0.5 * (logdet_f - logdet_s + quadratic_s - quadratic_f)

    #             scores = vmap(score)(ratios, means_f, chols_f, logdets_f)
    #             # Centre before logsumexp instead of subtracting two large
    #             # scalar results afterwards. A zero branch remains neutral.
    #             scores = scores - scores[index]
    #             return jnp.log(count) - logsumexp(scores)

    #         log_gamma = lax.map(jax.checkpoint(query_terms), (indices, q, means_q)).sum()
    #         return kl_qf, log_gamma

    #     kl_qf, log_gamma = lax.map(jax.checkpoint(factor_terms), messages)
    #     return kl_qf.sum(), log_gamma.sum(), kl_qp
