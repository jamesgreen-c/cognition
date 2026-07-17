import optax

import jax
import jax.random as jr
import jax.numpy as jnp

from tqdm import tqdm
from typing import Callable, Any

from jax import Array

from rp_ssm.free_energy import ConstrainedIVFreeEnergy
from rp_ssm.config import Config
from rp_ssm.distributions import AllParams

from rp_ssm.utils.linreg import linear_r2
from rp_ssm.utils.math import scale_sv, clip_sv


EPS = 1e-3


class Trainer:
    params: AllParams
    opt_states: list[optax.OptState]
    opts: list[optax.GradientTransformation]
    itr: int

    def __init__(
            self,
            free_energy: ConstrainedIVFreeEnergy,
            config: Config,
            logger: Callable = lambda *x: {}
    ):
        self.free_energy = free_energy
        self.config = config
        self.logger = logger
        self.itr = 0
        self.r2_history = []
        self.best_params = None
        self.best_loss = float('inf')

    def train_step(
            self,
            params: AllParams,
            opt_states: list[optax.OptState],
            data: tuple[Array]
    ) -> tuple[float, Any, AllParams, list[optax.OptState]]:
        beta = self.config.beta_schedule(self.itr)
        em = self.config.em
        
        (loss, aux), grads = jax.value_and_grad(
            self.free_energy.loss, has_aux=True
        )(params, data, beta, em)

        new_params, new_opt_states = (), ()
        for (param, grad, opt_state, opt) in zip(params, grads, opt_states, self.opts):

            # print(f'param: {param}\n\n, grad: {grad}\n\n, opt_state: {opt_state}\n\n, opt: {opt}\n\n')
            updates, new_opt_state = opt.update(grad, opt_state, param)
            new_param = optax.apply_updates(param, updates)
            new_params += (new_param,)
            new_opt_states += (new_opt_state,)

        return loss, aux, new_params, new_opt_states

    def fit(self, data: tuple[Array], use_pbar: bool = True, y: Array = None) -> None:
        N, T = data[0].shape[:2]
        print(f'Training with N={N}, T={T}')
        
        key, subkey = jr.split(jr.PRNGKey(self.config.seed))
        self.params, self.opt_states, self.opts = self.free_energy.init(subkey, data, self.config)

        train_step = jax.jit(self.train_step) if self.config.jit else self.train_step

        self.loss_tot = []
        self.r2_history = []

        self.best_params = None
        self.best_loss = float('inf')

        pbar = tqdm(range(self.config.num_iter), disable=not(use_pbar))
        for self.itr in pbar:
            key, subkey = jr.split(key)
            if self.config.batch_size == data[0].shape[0]:
                if self.itr == 0:
                    jax.debug.print('using entire dataset')
                batch_indices = jnp.arange(self.config.batch_size)
            else:
                batch_indices = jr.randint(subkey, (self.config.batch_size,), 0, data[0].shape[0])
            
            # print("Made it here")
            data_batch = [d[batch_indices] for d in data]

            # print("Made it here too")
            loss, aux, self.params, self.opt_states = train_step(
                self.params, self.opt_states, data_batch
            )
            
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = self.params
            
            self._stabilise_params()

            self.loss_tot.append(loss)
            to_print = self.logger(self, aux, batch_indices) # TODO: validation step?
            to_print.update({'loss': f'{loss:.3f}'})

            pbar.set_postfix(**to_print)

            if y is not None and self.itr % 100 == 0:
                x = self.apply((data[0], ))[1].params["means"]
                r2 = linear_r2(x, y)
                self.r2_history.append(r2)

    def train_continue(self, data: tuple[Array], new_iter: int, key: Array):
        train_step = jax.jit(self.train_step) if self.config.jit else self.train_step

        pbar = tqdm(range(self.itr, self.itr + new_iter))
        for self.itr in pbar:
            key, subkey = jr.split(key)
            batch_indices = jr.randint(subkey, (self.config.batch_size,), 0, data[0].shape[0])
            data_batch = [d[batch_indices] for d in data]

            loss, aux, self.params, self.opt_states = train_step(
                self.params, self.opt_states, data_batch
            )
            
            self._stabilise_params()

            self.loss_tot.append(loss)
            self.logger(self, aux, batch_indices) # TODO: validation step?
            R2s = ','.join(f'{r:.2f}' for r in self.r2[-1])
            pbar.set_postfix(loss=f'{loss:.3f}', R2=R2s)

    def _stabilise_params(self):
        if self.config.stabilise_A == 'scale':
            self.params[0]['A'] = scale_sv(self.params[0]['A'], EPS)
        elif self.config.stabilise_A == 'clip':
            self.params[0]['A'] = clip_sv(self.params[0]['A'], EPS)

    def apply(self, data: tuple[Array]) -> None:
        prior_params, *rec_params = self.params
        prior = self.free_energy.model.prior.update(prior_params)
        _, factors_nat, posterior = self.free_energy.get_posterior(prior, rec_params, data)
        factors = factors_nat.dist_param
        return factors, posterior
