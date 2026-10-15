"""
Single-chain Gaussian RAMP model; put in rp_slac/recognition/ramp.py.

There is one latent z_t per time, observation branches at each node, and
learned forward/backward messages on the chain. There is no LGSSM transition
density: the fixed node prior is f0 = N(0, I). Training lives in ramp_fe.py.

Recognition outputs are ratio natural parameters
r_j = eta(f_j) - eta(f0), so f_j has parameters prior + r_j.
Positive message precisions restrict the family, but ensure normalisability.

The parameter groups match rpm.py: 'rpm' contains observation recognisers;
'prior' contains the two message networks and the optional reward predictor,
not parameters of f0. Rewards supervise causal beliefs through an auxiliary
prediction loss; they are never inputs to the image encoder or messages.

Actions, when used, are observed leaves on the transition factors. Each
candidate subtree carries its own actions in the empirical mixtures.
This is a joint replay-data model of observations and actions.

Inference supplies Gaussian node marginals, not a joint trajectory posterior
or a generative transition kernel. Marginal sampling does not reproduce an
LGSSM path sampler.

Shape notation
--------------
B: number of sequences in the batch.
T: number of observation times.
J: number of observation recognisers.
N: latent_dim.
A: action_dim.
H: hidden_dim.
O_j: observation event shape for recogniser j.
F = N + N*(N+1)//2 + A: message-network input feature dimension.

GaussianNatParam shapes describe its 'pwm' vector and 'p' precision matrix.
GaussianDistParam shapes describe its 'mean' vector and 'cov' matrix.
Network parameters are Flax variable trees.
"""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr

from jax import Array, lax, vmap
from jax.tree_util import tree_map

from rp_slac.distributions import GaussianDistParam, GaussianNatParam
from rp_slac.recognition.rpm import GaussianRecognition
from rp_slac.config import Config


# class MessageFeatures(nn.Module):
#     """Nonlinear features; GaussianRecognition supplies the final head."""

#     hidden_dim: int = 128

#     @nn.compact
#     def __call__(self, x: Array) -> Array:
#         return nn.tanh(nn.Dense(self.hidden_dim)(x))


# class RewardPredictor(nn.Module):
#     """Gaussian p(r_t | z_t, a_t, z_{t+1}); no image reconstruction."""

#     features: Sequence[int] = (256, 256)
#     min_std: float = 1e-3

#     @nn.compact
#     def __call__(self, inputs: Array):
#         x = inputs

#         for width in self.features:
#             x = nn.leaky_relu(nn.Dense(width)(x))
#         x = nn.Dense(2)(x)

#         return x[..., 0], nn.softplus(x[..., 1]) + self.min_std


class RAMPSSM:
    """
    Time-shared messages on one SSM-shaped factor tree.

    Observation arrays share batch and time axes. Each node receives its local
    observation messages and messages from the two neighbouring nodes.
    Absent boundary edges send zero natural parameters.

    Parameters
    ----------
    latent_dim:  Latent dimension N.
    recognition: Sequence of length J. GaussianRecognition modules mapping 
                    observations with event shape O_j to ratio parameters pwm=(N,), p=(N,N).
    action_dim:  Edge-action dimension A; zero if unused.
    hidden_dim:  Message-network hidden dimension H.
    """

    def __init__(
        self,
        latent_dim: int,
        recognition: Sequence[GaussianRecognition],
        message_network: GaussianRecognition,
        reward_network: GaussianRecognition | None = None,
        action_dim: int = 0,
    ):
        if latent_dim < 1 or action_dim < 0 or not recognition:
            raise ValueError('Use positive dimensions and at least one recogniser.')
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.recognition = tuple(recognition)
        self.message_network = message_network
        self.reward_network = reward_network
        self.prior = GaussianNatParam(p=jnp.eye(latent_dim), pwm=jnp.zeros(latent_dim))

    def _data(self, data) -> tuple[Array, ...]:
        data = tuple(data) if isinstance(data, (tuple, list)) else (data,)

        if len(data) != len(self.recognition):
            raise ValueError('Supply one observation array per recogniser.')
        if any(x.ndim < 2 for x in data):
            raise ValueError('Observation arrays need batch and time axes.')
        
        shape = data[0].shape[:2]
        if min(shape) < 1 or any(x.shape[:2] != shape for x in data):
            raise ValueError('Observation arrays must share nonempty (B,T) axes.')
        
        return data

    def _actions(self, actions, batch_size, num_edges, dtype) -> Array:

        if actions is None and self.action_dim == 0:
            return jnp.zeros((batch_size, num_edges, 0), dtype=dtype)
        if actions is None:
            raise ValueError('Supply the recorded edge actions for this model.')
        if actions.ndim == 2 and self.action_dim == 1:
            actions = actions[..., None]
        if actions.shape != (batch_size, num_edges, self.action_dim):
            raise ValueError('Expected actions shaped (B,T-1,action_dim).')
        
        return actions.astype(dtype)

    def _message(self, params, ratio: GaussianNatParam, action: Array):
        """
        Encode a batch of cavity ratios and apply the existing Gaussian head.

        The prior is implicit and fixed. The cavity belief is prior + ratio.
        Only the precision's lower triangle is needed because it is symmetric.
        DistMap acts on one example, hence the vmap over the batch.

        Use arcsinh transformation as a smooth compressor for large natural parameters;
            recorded actions retain their original scale.
        
        Parameters
        ----------
        params: Parameters of one directed message network.
        ratio:  Cavity ratio with pwm=(B,N), p=(B,N,N).
        action: Actions on the outgoing edge, including (B,0) when unused.

        Returns
        -------
        output: Outgoing ratios with pwm=(B,N), p=(B,N,N). 
                    The intermediate network features have shape (B,F).
        """
        rows, cols = jnp.tril_indices(self.latent_dim)
        features = jnp.concatenate((
            jnp.arcsinh(ratio.params['pwm']),
            jnp.arcsinh(ratio.params['p'][..., rows, cols]), 
            action
        ), axis=-1)
        return vmap(lambda x: self.message_network.apply(params, x))(features)

    def init(self, key: Array, data, config: Config) -> dict:
        """Use the same model-init signature and optimizer groups as RPSSM."""

        data = self._data(data)
        count = len(self.recognition)
        keys = jr.split(key, count + 2 + (self.reward_network is not None))

        # encoder params
        rec_params = [
            rec.init(k, x[0, 0])
            for rec, k, x in zip(self.recognition, keys[:count], data)
        ]

        # nodewise rpm params
        input_dim = self.latent_dim + self.latent_dim * (self.latent_dim + 1) // 2
        dummy = jnp.zeros(input_dim + self.action_dim)
        prior_params = {
            'forward': self.message_network.init(keys[count], dummy),
            'backward': self.message_network.init(keys[count + 1], dummy),
        }

        # optional additional params for reward prediction if used in RAMPSLAC
        if self.reward_network is not None:
            reward_inputs = jnp.zeros(2 * self.latent_dim + self.action_dim)
            prior_params['reward'] = self.reward_network.init(keys[count + 2], reward_inputs)
        
        return {'prior': prior_params, 'rpm': rec_params}

    def get_factors(self, rec_params, data, return_features: bool = False):
        """
        Apply the observation recognisers over batch and time.

        Parameters
        ----------
        rec_params: (J,). Observation-network Flax variable trees, one per recogniser.
        data:       (J, ...) or (...) if J=1. Observation arrays shaped (B,T,*O_j).

        Returns
        -------
        output: Observation ratios with pwm=(J,B,T,N), p=(J,B,T,N,N).
        """
        data = self._data(data)

        if len(rec_params) != len(self.recognition):
            raise ValueError('Recognition parameter count does not match the model.')

        # apply recognition to data and remove features if not required
        outputs = [
            vmap(vmap(lambda x: rec.apply(p, x, return_features=return_features)))(datapoint)
            for rec, p, datapoint in zip(self.recognition, rec_params, data)
        ]
        outs = [out[0] for out in outputs] if return_features else outputs

        # stack outputted statistics into a single NatParam for all factors
        factors = type(outs[0])(
            **{k: jnp.stack([out.params[k] for out in outs]) 
            for k in outs[0].params.keys()}
        )

        if factors.latent_dim != self.latent_dim:
            raise ValueError('Recognition and message latent dimensions must match.')
        return (factors, outputs[0][1]) if return_features else factors

    def get_features(self, rec_params, observations: Array) -> Array:
        """Encode (B,T,...) frames with the first observation recogniser."""
        # TODO this should loop over rec: (J,), data: (J,)
        rec = self.recognition[0]
        _apply = lambda frame: rec.apply(rec_params[0], frame, method=rec.encode)
        return vmap(vmap(_apply))(observations)

    def predict_rewards(self, prior_params, latents: Array, actions: Array):
        """Predict rewards from causal latent pairs and recorded edge actions."""

        if self.reward_network is None:
            raise ValueError('This RAMP model has no reward predictor.')

        # reward predictor requires actions and transitions
        inputs = jnp.concatenate((latents[:, :-1], actions, latents[:, 1:]), axis=-1)
        _apply = lambda inp: self.reward_network.apply(prior_params["reward"], inp)
        return vmap(vmap(_apply))(inputs)

    def _pass(self, params, local, actions, reverse=False) -> GaussianNatParam:
        """
        Send messages along the chain without feeding a message back to its sender.

        At each source node the cavity is its observation ratio plus the message
        from the other side. The destination's message is excluded.

        Parameters
        ----------
        params:   Parameters of the forward or backward message network.
        local:    Summed observation ratios with pwm=(B,T,N), p=(B,T,N,N).
        actions:  (B,T-1,A). Validated edge actions, including (B,T-1,0) when unused.
        reverse:  False for a forward scan, True for a backward scan.

        Returns
        -------
        output: GaussianNatParam
            Incoming chain ratios at their destination nodes, with pwm=(B,T,N), p=(B,T,N,N).
            The first time is zero for a forward pass.
            The last time is zero for a backward pass.
        """
        zero = tree_map(lambda x: jnp.zeros_like(x[:, 0]), local)
        source_slice = slice(1, None) if reverse else slice(None, -1)
        sites = tree_map(lambda x: jnp.swapaxes(x[:, source_slice], 0, 1), local)

        def step(incoming, inputs):
            site, action = inputs
            outgoing = self._message(params, incoming + site, action)
            return outgoing, outgoing

        _, outgoing = lax.scan(
            step, zero, (sites, jnp.swapaxes(actions, 0, 1)), reverse=reverse
        )
        outgoing = tree_map(lambda x: jnp.swapaxes(x, 0, 1), outgoing)

        def pad(message, empty):
            parts = (message, empty[:, None]) if reverse else (empty[:, None], message)
            return jnp.concatenate(parts, axis=1)

        return tree_map(pad, outgoing, zero)

    def get_messages(self, prior_params, factors_nat, actions=None) -> GaussianNatParam:
        """
        Stack observation, left and right ratios at every latent node.

        The penultimate branch contains messages from the past.
        The final branch contains messages from the future.
        Their absent endpoint messages are exactly zero natural parameters.

        Parameters
        ----------
        prior_params:  Model params. 'forward' and 'backward' message-network parameters.
        factors_nat:   Observation ratios with pwm=(J,B,T,N), p=(J,B,T,N,N).
        actions:       (B,T-1,A), (B,T-1) when A=1, or None when A=0. Recorded edge actions.

        Returns
        -------
        output: GaussianNatParam
            All incoming ratios with pwm=(J+2,B,T,N), p=(J+2,B,T,N,N).
            The first J branches are observations, followed by left and right.
        """
        local = factors_nat.sum(axis=0)
        batch_size, timesteps = local.params['pwm'].shape[:2]
        actions = self._actions(actions, batch_size, timesteps - 1, local.params['pwm'].dtype)
        left = self._pass(prior_params['forward'], local, actions)
        right = self._pass(prior_params['backward'], local, actions, reverse=True)
        return tree_map(
            lambda obs, past, future: jnp.concatenate((obs, past[None], future[None]), axis=0),
            factors_nat, left, right,
        )

    def get_causal_messages(self, prior_params, factors_nat, actions=None) -> GaussianNatParam:
        """
        Stack observation and left ratios at every latent node.
        For use in apply() methods only requiring the forward causal beliefs.

        Parameters
        ----------
        prior_params:  Model params. 'forward' and 'backward' message-network parameters.
        factors_nat:   Observation ratios with pwm=(J,B,T,N), p=(J,B,T,N,N).
        actions:       (B,T-1,A), (B,T-1) when A=1, or None when A=0. Recorded edge actions.

        Returns
        -------
        output: GaussianNatParam
            All incoming causal ratios with pwm=(J+1,B,T,N), p=(J+1,B,T,N,N).
            The first J branches are observations, followed by left.
        """
        local = factors_nat.sum(axis=0)
        batch_size, timesteps = local.params['pwm'].shape[:2]
        actions = self._actions(actions, batch_size, timesteps - 1, local.params['pwm'].dtype)
        left = self._pass(prior_params['forward'], local, actions)
        return tree_map(
            lambda obs, past: jnp.concatenate((obs, past[None]), axis=0),
            factors_nat, left
        )

    # def _step_factors(self, rec_params, data) -> GaussianNatParam:
    #     """
    #     Sum the observation ratios for one online step.

    #     Parameters
    #     ----------
    #     rec_params: sequence of length J. Observation-network Flax variable trees.
    #     data:       sequence of length J, or one array when J=1. 
    #                     Current observation arrays shaped (B,*O_j), without a time axis.

    #     Returns
    #     -------
    #     output: GaussianNatParam
    #         Summed local ratio with pwm=(B,N), p=(B,N,N).
    #     """
    #     data = tuple(data) if isinstance(data, (tuple, list)) else (data,)
    #     # Reuse get_factors by adding a length-one time axis, then remove it.
    #     factors = self.get_factors(rec_params, tuple(x[:, None] for x in data))
    #     return tree_map(lambda x: x[:, 0], factors.sum(axis=0))

    # def initial_distribution(self, params: dict, data) -> GaussianDistParam:
    #     """First causal belief: fixed prior plus current observation ratios."""
    #     local = self._step_factors(params['rpm'], data)
    #     return (self.prior + local).dist_param

    # def filter(self, params: dict, posterior: GaussianDistParam, action, data):
    #     """
    #     Advance a filtered marginal with the batch-inference forward network.

    #     posterior is the previous returned belief; reset at episode boundaries.
    #     action is on the edge from that state to the current observation.

    #     Parameters
    #     ----------
    #     params:     Model parameters.
    #                     'prior': forward/backward message-network Flax variable trees.
    #                     'rpm': sequence of J observation-network Flax variable trees.
    #     posterior:  Previous filtered belief with mean=(B,N), cov=(B,N,N).
    #     action:     (B,A), (B,) when A=1, or None when A=0.
    #                     Action on the edge leading to the new observation.
    #     data:       (J, ...) or (...) if J=1. New observation arrays shaped (B,*O_j).

    #     Returns
    #     -------
    #     output: GaussianDistParam
    #         Next filtered belief with mean=(B,N), cov=(B,N,N).
    #     """
    #     ratio = posterior.nat_param - self.prior
    #     mean = posterior.params['mean']

    #     if action is not None:
    #         if action.ndim == 1 and self.action_dim == 1:
    #             action = action[:, None]
    #         action = action[:, None, :]

    #     action = self._actions(action, mean.shape[0], 1, mean.dtype)[:, 0]
    #     predicted = self._message(params['prior']['forward'], ratio, action)
    #     local = self._step_factors(params['rpm'], data)

    #     return (self.prior + predicted + local).dist_param
