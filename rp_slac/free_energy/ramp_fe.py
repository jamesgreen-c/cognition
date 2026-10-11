"""
Nodewise interior variational free energy.

Put this file in rp_slac/free_energy/ramp_fe.py.

This plays the role of model_fe.py for RAMPSSM, using the existing .kl(),
.lognormalizer and natural/distribution parameter APIs. Each time node has
its own RPM and contributes one prior KL plus all incoming message terms.
The batch approximates the full-dataset empirical subtree mixtures.

For incoming ratio r_j, define
G_j[n,m] = Phi(q_n+r_jm) - Phi(f0+r_jm).

The interior objective at one node and sequence is

    sum_j {G_j[n,n] - logmeanexp_m G_j[n,m] - KL(q_n || q_n+r_jn)} - KL(q_n || f0).

The auxiliary Gaussian q_n+r_jn is used to evaluate the bound; it does not
replace the inferred posterior. Parameter-independent empirical data masses
are omitted. beta=1 gives the bound; other values give its annealed variant.

Shape notation
--------------
B: number of sequences in the supplied batch.
T: observation times per sequence.
J: number of observation recognisers.
K = J+2: total incoming message branches.
N: latent_dim.
A: action_dim.
O_j: observation event shape for recogniser j.

GaussianNatParam shapes refer to 'pwm' and 'p'.
GaussianDistParam shapes refer to 'mean' and 'cov'.
Scalar JAX outputs have shape ().
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax

from jax import Array, lax, vmap
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

from rp_slac.config import Config
from rp_slac.distributions import AllParam
from rp_slac.recognition.ramp import RAMPSSM


class ConstrainedIVFreeEnergy:
    """Match the init/loss/get_posterior/get_loss_terms interface of model_fe.py."""

    def __init__(self, model: RAMPSSM):
        self.model = model

    def configure(self, config: Config):
        self.num_timesteps = config.sequence_length + 1
        self.batch_size = config.batch_size
        self.num_buffers = config.num_buffers
        self.num_factors = len(self.model.recognition) + 2

    def init(
        self, key: Array, data: tuple[Array, ...], config: Config
    ) -> tuple[dict, dict[str, optax.OptState], dict[str, optax.GradientTransformation]]:
        self.configure(config)
        observations, *_ = data
        params = self.model.init(key, observations, config)
        opts = {'prior': config.prior.build(), 'rpm': config.recognition.build()}
        opt_states = {name: opts[name].init(params[name]) for name in opts}
        return params, opt_states, opts

    def loss(
        self, params: dict, observations: Array, actions: Array | None,
        beta: float = 1.0, em: bool = False,
    ) -> tuple[Array, Any]:
        prior, factors_nat, posterior, filter_posterior = self.get_posterior(
            params['prior'], params['rpm'], actions, observations
        )

        # In EM mode hold the E-step belief fixed, but still train the messages.
        posterior = lax.cond(em, lax.stop_gradient, lambda q: q, posterior)
        kl_qf, log_Gamma, kl_qp = self.get_loss_terms(prior, factors_nat, posterior)

        batch_size, timesteps = posterior.params['mean'].shape[:2]
        Z = batch_size * timesteps  # One objective per node, not per incoming edge.
        loss = -(log_Gamma - kl_qf - beta * kl_qp) / Z
        return loss, {
            'posterior': posterior,
            'filter_posterior': filter_posterior,
            'factors_nat': factors_nat,  # Includes observation and both chain messages.
            'kl_qp': kl_qp / Z,
            'kl_qf': kl_qf / Z,
            'log_Gamma': log_Gamma / Z,
        }

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
        """
        factors_nat = self.model.get_factors(rec_params, observations)
        messages = self.model.get_messages(prior_params, factors_nat, actions)
        prior = self.model.prior
        posterior = (prior + messages.sum(axis=0)).dist_param
        left = tree_map(lambda x: x[-2], messages)
        filtered = (prior + factors_nat.sum(axis=0) + left).dist_param
        return prior, messages, posterior, filtered

    def get_loss_terms(self, prior, factors_nat, posterior):
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
        messages_by_time = tree_map(lambda x: jnp.moveaxis(x, 2, 0), factors_nat)
        q_by_time = tree_map(lambda x: jnp.swapaxes(x, 0, 1), q)

        # Map over time to avoid materialising all T batches of pairwise
        # Gaussian products. Rematerialise them during the backward pass.
        def node_terms(inputs):
            messages, node_q = inputs
            return self._node_loss_terms(prior, messages, node_q)

        terms = lax.map(jax.checkpoint(node_terms), (messages_by_time, q_by_time))
        return tuple(term.sum() for term in terms)

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
