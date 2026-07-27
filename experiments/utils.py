import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey

from rp_ssm.utils.dataset import Dataset



# def mix_data(key: PRNGKey, data: Dataset, new_data: Dataset):
#     """
#     Randomly mixes old and newly generated trajectories

#     Half of each split is sampled from the existing dataset and half from the
#     newly generated dataset. The resulting train and validation splits retain
#     their original sizes.

#     Parameters
#     ----------
#     key:       PRNG key used to sample and shuffle trajectories
#     data:      existing Dataset containing train and validation trajectories
#     new_data:  newly generated Dataset containing train and validation trajectories

#     Returns
#     -------
#     data: Dataset containing the mixed train and validation trajectories
#     """
#     train_key, val_key = jr.split(key)

#     def _mix_split(key, old_states, old_obs, old_actions, new_states, new_obs, new_actions):
#         old_key, new_key, shuffle_key = jr.split(key, 3)
#         N = old_states.shape[0]
#         N_new = N // 2
#         N_old = N - N_new

#         old_indices = jr.choice(
#             old_key, old_states.shape[0], shape=(N_old,), replace=False
#         )
#         new_indices = jr.choice(
#             new_key, new_states.shape[0], shape=(N_new,),
#             replace=new_states.shape[0] < N_new
#         )

#         states = jnp.concatenate([old_states[old_indices], new_states[new_indices]], axis=0)
#         observations = jnp.concatenate([old_obs[old_indices], new_obs[new_indices]], axis=0)
#         actions = jnp.concatenate([old_actions[old_indices], new_actions[new_indices]], axis=0)

#         indices = jr.permutation(shuffle_key, N)
#         return states[indices], observations[indices], actions[indices]

#     train_states, train_data, train_actions = _mix_split(
#         train_key,
#         data.train_states,
#         data.train_data[0],
#         data.params["train_actions"],
#         new_data.train_states,
#         new_data.train_data[0],
#         new_data.params["train_actions"]
#     )
#     val_states, val_data, val_actions = _mix_split(
#         val_key,
#         data.val_states,
#         data.val_data[0],
#         data.params["val_actions"],
#         new_data.val_states,
#         new_data.val_data[0],
#         new_data.params["val_actions"]
#     )

#     return Dataset(
#         train_data=(train_data,),
#         train_states=train_states,
#         val_data=(val_data,),
#         val_states=val_states,
#         params={
#             "train_actions": train_actions,
#             "val_actions": val_actions,
#         },
#     )

def mix_data(
        key: PRNGKey,
        anchor_set: Dataset,
        old_data: Dataset,
        new_data: Dataset,
        anchor_fraction: float = 0.5,
        old_fraction: float = 0.25,
    ):
    """
    Construct a fixed-size replay dataset containing permanent anchor data,
    previous replay data and newly generated trajectories.

    Validation data remains fixed to the anchor validation set.

    Parameters
    ----------
    key:              PRNG key used to sample and shuffle trajectories
    anchor_set:       Initial random-policy dataset retained permanently
    old_data:         Dataset used during the previous outer iteration
    new_data:         Newly generated policy dataset
    anchor_fraction:  Fraction sampled from the permanent anchor dataset
    old_fraction:     Fraction sampled from the previous replay dataset

    Returns
    -------
    data: Fixed-size Dataset containing mixed training trajectories and the
        original anchor validation split
    """
    anchor_key, old_key, new_key, shuffle_key = jr.split(key, 4)

    N = anchor_set.train_states.shape[0]
    N_anchor = int(N * anchor_fraction)
    N_old = int(N * old_fraction)
    N_new = N - N_anchor - N_old

    anchor_indices = jr.choice(
        anchor_key,
        anchor_set.train_states.shape[0],
        shape=(N_anchor,),
        replace=anchor_set.train_states.shape[0] < N_anchor,
    )
    old_indices = jr.choice(
        old_key,
        old_data.train_states.shape[0],
        shape=(N_old,),
        replace=old_data.train_states.shape[0] < N_old,
    )
    new_indices = jr.choice(
        new_key,
        new_data.train_states.shape[0],
        shape=(N_new,),
        replace=new_data.train_states.shape[0] < N_new,
    )

    train_states = jnp.concatenate([
        anchor_set.train_states[anchor_indices],
        old_data.train_states[old_indices],
        new_data.train_states[new_indices],
    ], axis=0)
    train_data = jnp.concatenate([
        anchor_set.train_data[0][anchor_indices],
        old_data.train_data[0][old_indices],
        new_data.train_data[0][new_indices],
    ], axis=0)
    train_actions = jnp.concatenate([
        anchor_set.params["train_actions"][anchor_indices],
        old_data.params["train_actions"][old_indices],
        new_data.params["train_actions"][new_indices],
    ], axis=0)

    indices = jr.permutation(shuffle_key, N)

    return Dataset(
        train_data=(train_data[indices],),
        train_states=train_states[indices],
        val_data=anchor_set.val_data,
        val_states=anchor_set.val_states,
        params={
            "train_actions": train_actions[indices],
            "val_actions": anchor_set.params["val_actions"],
        },
    )