import jax.numpy as jnp

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
        env_dim: int,
        batch_size: int,
        num_buffers: int,
        pretrain_iter: int,
        num_iter: int,
        gamma: float,
        seed: int,
        stabilise_A: str | None = None,
):
    D = latent_dim
    K = env_dim

    # CONFIG
    CFG = Config(
        sequence_length=sequence_length,
        num_pretrain=pretrain_iter,
        num_iter=num_iter,
        batch_size=batch_size,
        num_buffers=num_buffers,
        gamma=gamma,
        jit=True,
        stabilise_A=stabilise_A,
        seed=seed,
    )

    # MODEL DEFINITION
    A = jnp.zeros((D, D))
    B = jnp.ones((D, K))
    PRIOR = distributions.LGStationaryParam(stationary=True, A=A, B=B)
    REC = rpm.GaussianRecognition(
        network=networks.MLP([32, 32, 32]),
        dist_map=distmaps.MVNDiag(D),
        constant_cov=True
    )
    MODEL = rpm.RPSSM(prior=PRIOR, recognition=(REC, ))
    MODEL_FE = ConstrainedIVFreeEnergy(model=MODEL)

    # CONTROL DEFINITION
    actor_net = actor.ActorNetwork(
        network=networks.MLP([32, 32, 32]),
        dist_map=distmaps.MVNDiag(K)
    )
    ACTOR = actor.Actor(network=actor_net)

    critic_net = critic.CriticNetwork(network=networks.MLP([32, 32, 32]))
    CRITIC = critic.Critic(network=critic_net)
    CONTROL_FE = ControlFreeEnergy(actor=ACTOR, critic=CRITIC)

    return CFG, MODEL_FE, CONTROL_FE
