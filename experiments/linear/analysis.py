import os
import pickle
import argparse
import matplotlib.pyplot as plt 

from rp_ssm import training
from rp_ssm.recognition import networks
from rp_ssm.utils.nnreg import DecoderNetwork, DecoderTrainer, Config as DecConfig

from experiments.linear.config import setup

# ARGS PARSING
parser = argparse.ArgumentParser()
parser.add_argument("--D", dest="D", type=int, default=5)

parser.add_argument("--N", dest="N", type=int, default=250)
parser.add_argument("--T", dest="T", type=int, default=100)

parser.add_argument("--stabilise", dest="stabilise", default="clip")

parser.add_argument("--num-iter", dest="num_iter", type=int, default=500)
parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)

parser.add_argument("--seed", dest="seed", type=int, default=1234)

parser.add_argument("--regression", action="store_true")
parser.add_argument("--no-regression", dest="regression", action="store_false")
parser.set_defaults(regression=False)

parser.add_argument("--B", dest="B", type=int, default=10)

args = parser.parse_args()

CFG, _, _, _, FREE_ENERGY = setup(args.D, args.batch_size, args.num_iter, args.seed, args.stabilise)
FREE_ENERGY.num_timesteps = args.T

# LOAD SAVED DATA, PARAMS and LOSS
EXPERIMENT_NAME = f"D={args.D},N={args.N},T={args.T},iter={args.num_iter},batch-size={args.batch_size},stabilise={args.stabilise},seed={args.seed}"
DIRPATH = f"results/{EXPERIMENT_NAME}"

if not os.path.exists(DIRPATH):
    raise FileNotFoundError(f"No saved parameter results for experiment: {EXPERIMENT_NAME}")

with open(f"{DIRPATH}/params.pkl", "rb") as f:
    PARAMS = pickle.load(f)

with open(f"{DIRPATH}/loss.pkl", "rb") as f:
    LOSS = pickle.load(f)

with open(f"{DIRPATH}/data.pkl", "rb") as f:
    DATA = pickle.load(f)
    DATA = DATA.standardised_data


# RECONSTRUCT TRAINER
TRAINER = training.Trainer(free_energy=FREE_ENERGY, config=CFG)
TRAINER.params = PARAMS

PLOTDIR = f"{DIRPATH}/plots"
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

def train_decoder():
    """ train the decoder on RPM posterior means to ground truths """
    true_dim = DATA.train_states[0].shape[-1]

    # apply model to get posterior train state means
    _, posterior = TRAINER.apply((DATA.train_data[0], ))
    means = posterior.params["means"]

    # define and fit decoder
    net = DecoderNetwork(network=networks.MLP([10, 10, 10]), output_dim=true_dim)
    decfg = DecConfig(batch_size=64, iterations=3000, seed=args.seed+2)
    decoder = DecoderTrainer(model=net, config=decfg)
    decoder.fit(x=DATA.train_states, y=means)

    # # check fit
    preds = decoder.apply(params=decoder.params, y=means)
    return decoder, preds


def apply_decoder(decoder: DecoderTrainer):
    """ apply learned decoder to RPM posterior means on validation data """
    _slice = slice(args.B)  # number of val sequences to test against

    # apply model to get posterior validation state means
    _, posterior = TRAINER.apply((DATA.val_data[0][_slice], ))
    means = posterior.params["means"]

    # apply trained decoder
    return decoder.apply(params=decoder.params, y=means)
    

def analyse_regression():
    """
    Plot:
    1. The loss of the decoder training over time
    2. The regression for the RPM train state means to the true train latents
    3. The regression for the RPM validation state means to the true validation latents
    """
    decoder, training_preds = train_decoder()
    validation_preds = apply_decoder(decoder)  # (B, T, true_dim)

    true_dim = DATA.train_states[0].shape[-1]

    # plot training examples (same number of validation examples)
    fig, ax = plt.subplots(args.B, true_dim, figsize=(15, 5*args.B))
    for b in range(args.B):
        for d in range(true_dim):
            ax[b, d].plot(training_preds[b, :, d])
            ax[b, d].plot(DATA.train_states[b, :, d], color="black", linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{PLOTDIR}/train_regression.png")

    # plot validations examples
    fig, ax = plt.subplots(args.B, true_dim, figsize=(15, 5*args.B))
    for b in range(args.B):
        for d in range(true_dim):
            ax[b, d].plot(validation_preds[b, :, d])
            ax[b, d].plot(DATA.val_states[b, :, d], color="black", linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{PLOTDIR}/validation_regression.png")

    # plot decoder loss
    plt.figure(figsize=(15, 5))
    plt.plot(decoder.loss_tot)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss over training iterations")
    plt.tight_layout()
    plt.savefig(f"{PLOTDIR}/decoder_loss.png")
    plt.close()


def analyse_training():
    """
    Plot:
    1. Loss over time
    2. Learned A matrix
    """

    # plot loss
    plt.figure(figsize=(15, 5))
    plt.plot(LOSS)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss over training iterations")
    plt.tight_layout()
    plt.savefig(f"{PLOTDIR}/loss.png")
    plt.close()

    A = PARAMS[0]["A"]  # (D, D)
    plt.figure()
    plt.imshow(A)
    plt.title("Learned A on representation")
    plt.tight_layout()
    plt.savefig(f"{PLOTDIR}/A.png")


if __name__ == "__main__":

    analyse_training()
    
    if args.regression:
        analyse_regression()
    
