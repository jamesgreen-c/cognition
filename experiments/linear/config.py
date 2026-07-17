import jax.numpy as jnp

from rp_ssm import distributions, config
from rp_ssm.free_energy import ConstrainedIVFreeEnergy
from rp_ssm.recognition import distmaps, networks, rpm


def setup(
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
        network=networks.MLP([5, 5]),
        dist_map=distmaps.MVNDiag(D),
        constant_cov=True
    )
    MODEL = rpm.RPSSM(prior=PRIOR, recognition=(REC, ))
    FREE_ENERGY = ConstrainedIVFreeEnergy(model=MODEL)

    return CFG, PRIOR, REC, MODEL, FREE_ENERGY