import jax.numpy as jnp

from jax import Array

from rp_slac import actor
from rp_slac import critic
from rp_slac.config import Config
from rp_slac import distributions

from rp_slac.recognition import networks, distmaps
from rp_slac.free_energy import model_fe, ramp_fe
from rp_slac.free_energy.control_fe import ControlFreeEnergy
from rp_slac.recognition import rpm, ramp 


def setup(
    sequence_length: int,
    initial_steps: int,
    latent_dim: int,
    actor_history: int,
    action_dim: int,
    action_low: Array,
    action_high: Array,
    model_batch_size: int,
    sac_batch_size: int,
    pixels: int,
    num_buffers: int,
    pretrain_iter: int,
    num_iter: int,
    capacity: int,
    gamma: float,
    seed: int,
    kernel: str = "ramp",
):
    if kernel not in ("ramp", "rpssm"):
        raise ValueError("kernel must be 'ramp' or 'rpssm'.")
    
    D = latent_dim
    K = action_dim

    CFG = Config(
        sequence_length=sequence_length,
        num_pretrain=pretrain_iter,
        num_iter=num_iter,
        model_batch_size=model_batch_size,
        sac_batch_size=sac_batch_size,
        num_buffers=num_buffers,
        actor_history=actor_history,
        initial_steps=initial_steps,
        capacity=capacity,
        gamma=gamma,
        actor_state="observation",
        jit=True,
        stabilise_A="scale" if kernel == "rpssm" else None,
        seed=seed,
        target_update_rate=0.005
    )

    # learning rates
    CFG.prior.lr = 1e-3
    CFG.recognition.lr = 1e-3
    CFG.actor.lr = 1e-3
    CFG.critic.lr = 1e-3
    CFG.alpha.lr = 1e-3

    # recognition definition
    last_kernel = (pixels + 15) // 16
    cnn_features = [
        {"features": 32, "kernel_size": (5, 5), "strides": (2, 2), "padding": "SAME"},
        {"features": 64, "kernel_size": (3, 3), "strides": (2, 2), "padding": "SAME"},
        {"features": 128, "kernel_size": (3, 3), "strides": (2, 2), "padding": "SAME"},
        {"features": 256, "kernel_size": (3, 3), "strides": (2, 2), "padding": "SAME"},
        {"features": 256, "kernel_size": (last_kernel, last_kernel), "strides": (1, 1), "padding": "VALID"},
    ]

    REC = rpm.GaussianRecognition(
        compressor=networks.DenseCNN(cnn_features=cnn_features),
        network=networks.MLP([256, 256]),
        dist_map=distmaps.MVNDiag(D, max_precision=1000.0),
        constant_cov=True,
    )
    REWARD_NET = rpm.GaussianRecognition(
        network=networks.MLP([256, 256]),
        dist_map=distmaps.MVNDiag(1),
    )

    # model definition
    if kernel == "ramp":
        RPM = rpm.GaussianRecognition(
            network=networks.TanhMLP([256, 256, 256, 256]),
            dist_map=distmaps.MVNCholeskySoftplus(D, max_chol=10.0),
        )
        MODEL = ramp.RAMPSSM(
            latent_dim=D,
            recognition=(REC,),
            message_network=RPM,
            action_dim=K,
            reward_network=REWARD_NET,
        )
        MODEL_FE = ramp_fe.ConstrainedIVFreeEnergy(model=MODEL)

    else:
        # Match the existing RPSSM prior initialisation.
        PRIOR = distributions.LGStationaryParam(
            stationary=True, 
            A=jnp.zeros((D, D)), 
            B=jnp.ones((D, K)),
        )
        MODEL = rpm.RPSSM(prior=PRIOR, recognition=(REC,), reward_network=REWARD_NET)
        MODEL_FE = model_fe.ConstrainedIVFreeEnergy(model=MODEL)

    # The policy consumes shared encoder features and preceding actions.
    actor_net = actor.ActorNetwork(
        network=networks.MLP((256, 256)),
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
    CONTROL_FE = ControlFreeEnergy(actor=ACTOR, critic=CRITIC, encoder=MODEL)

    return CFG, MODEL_FE, CONTROL_FE

