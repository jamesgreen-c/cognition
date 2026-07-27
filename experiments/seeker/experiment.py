import os
import pickle
import argparse

import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey

from rp_ssm.training import Trainer
from rp_ssm.utils.dataset import Dataset

from rp_control.control import ActorCritic

from experiments.seeker.data import get_data, random_policy, CentreSeekingEnvironment
from experiments.seeker.config import rpm_setup, ac_setup
from experiments.utils import mix_data


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# ARGS PARSING
parser = argparse.ArgumentParser()
parser.add_argument("--D", dest="D", type=int, default=8)

parser.add_argument("--N", dest="N", type=int, default=250)
parser.add_argument("--T", dest="T", type=int, default=100)

parser.add_argument("--stabilise", dest="stabilise", default="clip")

parser.add_argument("--num-iter", dest="num_iter", type=int, default=10)

parser.add_argument("--rec-steps", dest="rec_steps", type=int, default=500)
parser.add_argument("--rpm-batch-size", dest="rpm_batch_size", type=int, default=32)

parser.add_argument("--gamma", dest="gamma", type=float, default=0.99)
parser.add_argument("--control-steps", dest="control_steps", type=int, default=100)
parser.add_argument("--ac-batch-size", dest="ac_batch_size", type=int, default=32)

parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--ac-seed", dest="ac_seed", type=int, default=9876)

args = parser.parse_args()


# SETUP
RPM_PRETRAINING_ITER = 500
CFG, PRIOR, REC, MODEL, FREE_ENERGY = rpm_setup(args.D, args.rpm_batch_size, RPM_PRETRAINING_ITER, args.seed, args.stabilise)

ENV = CentreSeekingEnvironment(T=args.T)
AC_PRETRAINING_ITER = 200
AC_CFG, AC_LOSS = ac_setup(args.gamma, args.ac_batch_size, AC_PRETRAINING_ITER, args.ac_seed, MODEL, ENV)



def outer_step(key: PRNGKey, trainer: Trainer, actorcritic: ActorCritic, data: Dataset, anchor_data: Dataset):
    """
    
    """
    rpm_key, ac_key, gen_key, mix_key = jr.split(key, 4)

    # train RPM for some iters on dataset
    trainer.train_continue(data=data.standardised_data.train_data, new_iter=args.rec_steps, key=rpm_key)

    # train ActorCritic for some iters
    actorcritic.fit_continue(new_rpm_params=trainer.params, new_iter=args.control_steps, key=ac_key)

    # use learned actor policy to generate some new data
    new_data = actorcritic.generate(key=gen_key, N=args.N)  # needs to keep train and val structure

    # integrate new data with old data
    data = mix_data(mix_key, anchor_data, data, new_data)

    return trainer, actorcritic, data


def main(key: PRNGKey):

    key, data_key, gen_key, mix_key = jr.split(key, 4)

    # initial dataset
    anchor_data = get_data(
        key=data_key,
        policy=random_policy,
        num_factors=1,
        num_sequences=args.N,
        num_timesteps=args.T,    
    )
    data = anchor_data

        

    # train the RPSSM on initial dataset
    trainer = Trainer(free_energy=FREE_ENERGY, config=CFG)
    trainer.fit(data.standardised_data.train_data, use_pbar=True)

    # initial SLAC training
    actorcritic = ActorCritic(loss=AC_LOSS, config=AC_CFG, environment=ENV)
    actorcritic.fit(data.train_data, trainer.params)  # explicitly non-standardised data

    # generate next data
    new_data = actorcritic.generate(key=gen_key, N=args.N)

    # integrate new data
    data = mix_data(mix_key, anchor_data, data, new_data)

    # run training loop
    for iter in range(args.num_iter):
        key, subkey = jr.split(key)
        trainer, actorcritic, data = outer_step(subkey, trainer, actorcritic, data, anchor_data)

    
    # save data, params, and loss
    experiment_name = f"D={args.D},N={args.N},T={args.T},iter={args.num_iter},stabilise={args.stabilise},seed={args.seed}"
    dirpath = f"results/{experiment_name}"
    if not os.path.exists(dirpath): os.makedirs(dirpath, exist_ok=True)

    with open(f"{dirpath}/params.pkl", "wb") as f: 
        pickle.dump(trainer.params, f)
    with open(f"{dirpath}/loss.pkl", "wb") as f: 
        pickle.dump(trainer.loss_tot, f)
    with open(f"{dirpath}/data.pkl", "wb") as f: 
        pickle.dump(data, f)
    with open(f"{dirpath}/actorcritic.pkl", "wb") as f:
        pickle.dump(actorcritic.params, f)
    with open(f"{dirpath}/rewards.pkl", "wb") as f:
        pickle.dump(actorcritic.reward_hist, f)

    return trainer


if __name__ == "__main__":
    key_ = PRNGKey(args.seed + 1)
    trainer = main(key_)
