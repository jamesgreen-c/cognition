import jax.numpy as jnp

from jax import Array

from rp_slac import actor
from rp_slac import critic
from rp_slac.config import Config
from rp_slac import distributions

from rp_slac.recognition import rpm, networks, distmaps
from rp_slac.free_energy.model_fe import ConstrainedIVFreeEnergy
from rp_slac.free_energy.control_fe import ControlFreeEnergy


def setup(
        sequence_length: int,
        latent_dim: int,
        action_dim: int,
        action_low: Array,
        action_high: Array,
        batch_size: int,
        num_buffers: int,
        pretrain_iter: int,
        num_iter: int,
        collection_steps: int,
        capacity: int,
        gamma: float,
        seed: int,
        stabilise_A: str | None = None,
):
    D = latent_dim
    K = action_dim

    CFG = Config(
        sequence_length=sequence_length,
        num_pretrain=pretrain_iter,
        num_iter=num_iter,
        batch_size=batch_size,
        num_buffers=num_buffers,
        collection_steps=collection_steps,
        capacity=capacity,
        gamma=gamma,
        actor_state="observation",
        jit=True,
        stabilise_A=stabilise_A,
        seed=seed,
    )

    # prior definition
    A = jnp.zeros((D, D))
    B = jnp.ones((D, K))
    PRIOR = distributions.LGStationaryParam(stationary=True, A=A, B=B)

    # recognition definition
    cnn_features = [
        {"features": 32, "kernel_size": (5, 5), "strides": (2, 2), "padding": "SAME"},
        {"features": 64, "kernel_size": (3, 3), "strides": (2, 2), "padding": "SAME"},
        {"features": 128, "kernel_size": (3, 3), "strides": (2, 2), "padding": "SAME"},
        {"features": 256, "kernel_size": (3, 3), "strides": (2, 2), "padding": "SAME"},
        {"features": 256, "kernel_size": (4, 4), "strides": (1, 1), "padding": "VALID"},
    ]

    REC = rpm.GaussianRecognition(
        network=networks.DenseCNN(cnn_features=cnn_features, mlp_features=(256, 256)),
        dist_map=distmaps.MVNDiag(D),
        constant_cov=True,
    )

    # model definition
    MODEL = rpm.RPSSM(prior=PRIOR, recognition=(REC,))
    MODEL_FE = ConstrainedIVFreeEnergy(model=MODEL)

    # observation-conditioned actor
    actor_net = actor.ActorNetwork(
        network=networks.DenseCNN(cnn_features=cnn_features, mlp_features=(256, 256)),
        dist_map=distmaps.SLACMVNDiag(K),
    )
    ACTOR = actor.Actor(
        network=actor_net,
        action_low=action_low,
        action_high=action_high,
    )

    # latent-action critic
    critic_net = critic.CriticNetwork(network=networks.MLP([256, 256]))
    CRITIC = critic.Critic(network=critic_net)
    CONTROL_FE = ControlFreeEnergy(actor=ACTOR, critic=CRITIC)

    return CFG, MODEL_FE, CONTROL_FE