"""

Write trainer like in RPM-SLAM, to retrieve the loss from a jitted free_energy function. 
Then run optax updates
"""
from dataclasses import dataclass, field
from typing import Callable, Union, Optional, Tuple
from tqdm import tqdm

import optax
import jax
from jax import Array
from jax import tree_util

import jax.random as jr
import jax.numpy as jnp
import flax.linen as nn

from rpm.utils import stability


EPS = 1e-3
LearningRate = Union[float, Callable[[int], float]]


@dataclass
class OptimConfig:
    """ Configuration for a single parameter block optimiser. """
    optimizer: Callable[[float], optax.GradientTransformation] = optax.adam
    lr: LearningRate = 1e-3

    decay_steps: Optional[int] = None
    decay_rate: Optional[float] = None
    staircase: bool = False

    def schedule(self) -> LearningRate:
        """
        Returns either a constant learning rate, a user-provided schedule, or an exponential-decay schedule.
        """
        if callable(self.lr):
            return self.lr

        if self.decay_steps is None or self.decay_rate is None:
            return self.lr

        return optax.exponential_decay(
            init_value=self.lr,
            transition_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=self.staircase,
        )

    def build(self) -> optax.GradientTransformation:
        """ Builds the Optax optimiser. """
        return self.optimizer(self.schedule())


@dataclass
class Config:
    """
    Training configuration for recognition and prior optimisation.
    """
    batch_size: int = 32
    num_iter: int = 1000
    seed: int = 0

    sample_with_replacement: bool = False
    beta_schedule: LearningRate = lambda i: 1.
    stabilize_A: Optional[str] = 'scale'
    em: bool = False  # if True, perform EM (don't backprop through posterior)

    debug: bool = False

    rec: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            optimizer=optax.adam,
            lr=1e-3,
        )
    )

    prior: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            optimizer=optax.adam,
            lr=1e-3,
        )
    )


class Trainer:

    def __init__(
            self,
            network: Tuple[nn.Module],
            free_energy,
            prior,
            # posterior_function: Callable,
            # loss_function: Callable,
            # prior_init: Callable,
            config: Config,
        ):
        """
        Parameters
        ----------
        network:             Recognition network or tuple of recognition networks.
        posterior_function:  Callable returning recognition factors and posterior samples.
        loss_function:       Callable returning the stochastic loss and next cached samples.
        prior_init:          Callable initialising the prior parameter dictionary.
        sample_init:         Callable initialising the latent reference paths/cache.
        config:              Training configuration.
        stabilise_function:  Optional callable applied to prior parameters after each update.
        """

        self.loss_hist = None

        self.network = tuple(network) if isinstance(network, (tuple, list)) else (network,)
        self.free_energy = free_energy
        self.prior = prior
        self.config = config

        self.rec_opt = config.rec.build()
        self.prior_opt = config.prior.build()

        self._train_data_ndim = None

    def train_step(
            self,
            key,
            params,
            opt_states,
            data: Union[Array, Tuple[Array]],
        ):
        """
        Runs a single stochastic gradient update.

        Parameters
        ----------
        key:        PRNG key used by the stochastic free-energy estimator.
        params:     Dictionary with fields:
                        - "rec": recognition parameters
                        - "prior": prior parameters
        opt_states: Dictionary with fields:
                        - "rec": recognition optimiser state
                        - "prior": prior optimiser state
        data:       Mini-batch of observations.

        Returns
        -------
        loss:            Scalar stochastic loss.
        new_params:      Updated parameter dictionary.
        new_opt_states:  Updated optimiser-state dictionary.
        """

        # get loss and gradients
        (loss, aux), grads = jax.value_and_grad(self.free_energy.loss, argnums=2, has_aux=True)(key, self.network, params, data)

        # collect updates
        rec_updates, rec_opt_state = self.rec_opt.update(grads["rec"], opt_states["rec"], params["rec"])
        prior_updates, prior_opt_state = self.prior_opt.update(grads["prior"]["trainable"], opt_states["prior"], params["prior"]["trainable"],)

        # update parameters and opt_states
        new_opt_states = {"rec": rec_opt_state, "prior": prior_opt_state}
        new_params = {
            "rec": optax.apply_updates(params["rec"], rec_updates),
            "prior": {
                "trainable": optax.apply_updates(params["prior"]["trainable"], prior_updates),
                "fixed": params["prior"]["fixed"],
            },
        }

        return loss, aux, new_params, new_opt_states


    def fit(self, data):
        """
        Runs consecutive training steps.

        Parameters
        ----------
        data:   Array (N, T, *_) observations of batch B independent timeseries of length T, and of unknown complexity

        Returns
        -------
        best_params:  Parameter dictionary achieving the lowest observed stochastic loss.
        """
       
        data_leaf = tree_util.tree_leaves(data)[0]
        N, T = data_leaf.shape[:2]
        self._train_data_ndim = data_leaf.ndim
        print(f"Training with N={N}, T={T}")

        train_step = jax.jit(self.train_step) if not self.config.debug else self.train_step

        # RNG keys
        key, prior_init_key = jr.split(jr.PRNGKey(self.config.seed), 2)
        rec_init_keys = jr.split(key, len(self.network))

        # initialisation
        example_obs = tree_util.tree_map(lambda z: z[0, 0], data)

        _rec_params = tuple(rec.init(_k, example_obs) for rec, _k in zip(self.network, rec_init_keys))
        _prior_params = self.prior_init(prior_init_key)

        self.params = {
            "rec": _rec_params,
            "prior": _prior_params,
        }
        self.opt_states = {
            "rec": self.rec_opt.init(self.params["rec"]),
            "prior": self.prior_opt.init(self.params["prior"]["trainable"]),
        }

        # stores
        self.best_params = None
        self.best_loss = float("inf")
        self.loss_hist = []

        # batching
        batch_size = min(N, self.config.batch_size)
        batch_idx = self.get_batching_function(batch_size, N)

        # run
        pbar = tqdm(range(self.config.num_iter))
        for self.itr in pbar:
            key, batch_key, sample_key = jr.split(key, 3)

            idx = batch_idx(batch_key)
            batch = tree_util.tree_map(lambda z: z[idx], data)

            loss, aux, self.params, self.opt_states = train_step(
                sample_key,
                self.params,
                self.opt_states,
                batch,
            )

            # track loss
            loss_float = float(loss) / batch_size
            self.loss_hist.append(loss_float)
            pbar.set_postfix(loss=f"{loss_float:.3f}")

            if loss_float < self.best_loss:
                self.best_loss = loss_float
                self.best_params = self.params

            # update params
            self.params = {
                **self.params,
                "prior": self._stabilise_params(self.params["prior"])
            }

        return self.best_params
    
    def get_batching_function(self, batch_size: int, N: int):
        """
        Constructs a minibatch index sampler.

        Parameters
        ----------
        batch_size: Number of time series per minibatch.
        N:          Total number of time series.

        Returns
        -------
        batch_idx:  Callable mapping an RNG key to minibatch indices.
        """
        if batch_size == N:
            print("Using entire dataset")
            batch_idx = lambda _: jnp.arange(N)
        else:
            if self.config.sample_with_replacement:
                batch_idx = lambda _k: jr.randint(_k, (batch_size,), 0, N)
            else:
                batch_idx = lambda _k: jr.choice(_k, N, shape=(batch_size,), replace=False)
        return batch_idx
    
    def _stabilise_params(self):
        if self.config.stabilize_A == 'scale':
            self.params["prior"]['A'] = stability.scale_sv(self.params[0]['A'], EPS)
        elif self.config.stabilize_A == 'clip':
            self.params["prior"]['A'] = stability.clip_sv(self.params[0]['A'], EPS)

    def apply(self, key, data):
        """
        Runs posterior sampling on new data.

        Parameters
        ----------
        key: RNG key used for initialisation and posterior sampling.
        data: Observation PyTree with leaves of shape (B, T, *_).

        Returns
        -------
        factors:   Recognition factors from the final posterior call.
        posterior: Posterior means and covs for the LGSSM.
        """
        _, factors, posterior = self.free_energy.get_posterior(self.params, data)
        return factors, posterior
