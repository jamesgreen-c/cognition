import jax.numpy as jnp

from rp_ssm import distributions, config
from rp_ssm.free_energy import ConstrainedIVFreeEnergy
from rp_ssm.recognition import distmaps, networks, rpm

from rp_control.config import Config as ActorCriticConfig
from rp_control.loss import ActorCriticLoss
from rp_control.actor import Actor, ActorNetwork
from rp_control.critic import Critic, CriticNetwork
from rp_control.environment import Environment


def rpm_setup(
        D: int,
        batch_size: int,
        num_iter: int,
        seed: int,
        stabilise_A: str | None = None,

):
    
    # CONFIG
    CFG = config.Config(
        prior_lr=1e-2,
        rec_lr=(1e-2,),
        num_iter=num_iter,
        batch_size=batch_size,
        jit=True,
        stabilise_A=stabilise_A,
        seed=seed,
    )

    # MODEL DEFINITION
    PRIOR = distributions.LGStationaryParam(stationary=True, A=jnp.zeros((D, D)))
    REC = rpm.GaussianRecognition(
        network=networks.MLP([32, 32, 32]),
        dist_map=distmaps.MVNDiag(D),
        constant_cov=True
    )
    MODEL = rpm.RPSSM(prior=PRIOR, recognition=(REC, ))
    FREE_ENERGY = ConstrainedIVFreeEnergy(model=MODEL)

    return CFG, PRIOR, REC, MODEL, FREE_ENERGY


def ac_setup(gamma: float, batch_size: int, num_iter: int, seed: int, model: rpm.RPSSM, environment: Environment):

    AC_CFG = ActorCriticConfig(
        gamma=gamma,
        num_iter=num_iter,
        batch_size=batch_size,
        seed=seed
    )

    # actor
    actor_net = ActorNetwork(
        network=networks.MLP([32, 32, 32]),
        dist_map=distmaps.MVNDiag(1)
    )
    ACTOR = Actor(network=actor_net, action_shape=())

    # critic
    critic_net = CriticNetwork(network=networks.MLP([32, 32, 32]))
    CRITIC = Critic(network=critic_net)

    # loss
    LOSS = ActorCriticLoss(
        actor=ACTOR,
        critic=CRITIC,
        rpm=model,
        environment=environment,
    )

    return AC_CFG, LOSS