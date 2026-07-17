import os
import pickle
import argparse

from jax.random import PRNGKey

from rp_ssm import training

from experiments.linear.data import get_data
from experiments.linear.config import setup

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# ARGS PARSING
parser = argparse.ArgumentParser()
parser.add_argument("--D", dest="D", type=int, default=5)

parser.add_argument("--N", dest="N", type=int, default=250)
parser.add_argument("--T", dest="T", type=int, default=100)

parser.add_argument("--stabilise", dest="stabilise", default="clip")

parser.add_argument("--num-iter", dest="num_iter", type=int, default=500)
parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)

parser.add_argument("--seed", dest="seed", type=int, default=1234)
args = parser.parse_args()


CFG, PRIOR, REC, MODEL, FREE_ENERGY = setup(args.D, args.batch_size, args.num_iter, args.seed, args.stabilise)

# DATA DIMENSIONALITY
LATENT_DIM = 3
EMISSION_DIM = 5

def main(key: PRNGKey):

    # data
    data = get_data(
        key=key,
        num_factors=1,
        latent_dim=LATENT_DIM, 
        emission_dim=EMISSION_DIM,
        num_sequences=args.N,
        num_timesteps=args.T,    
    )

    # train the RPSSM
    trainer = training.Trainer(free_energy=FREE_ENERGY, config=CFG)
    trainer.fit(data.standardised_data.train_data, use_pbar=True)

    # save data, params, and loss
    experiment_name = f"D={args.D},N={args.N},T={args.T},iter={args.num_iter},batch-size={args.batch_size},stabilise={args.stabilise},seed={args.seed}"
    dirpath = f"results/{experiment_name}"
    if not os.path.exists(dirpath): os.mkdir(dirpath)

    with open(f"{dirpath}/params.pkl", "wb") as f: 
        pickle.dump(trainer.params, f)
    with open(f"{dirpath}/loss.pkl", "wb") as f: 
        pickle.dump(trainer.loss_tot, f)
    with open(f"{dirpath}/data.pkl", "wb") as f: 
        pickle.dump(data, f)

    return trainer


if __name__ == "__main__":
    key_ = PRNGKey(args.seed + 1)
    trainer = main(key_)

    # def main():
    # ...
    # # train the RPSSM and standardise the data
    # data, trainer, free_energy, rp_ssm = train_rpssm(
    #     dataset_str=dataset_str,
    #     prior=prior,
    #     dist_map=dist_map,
    #     network=network,
    #     save=False,
    #     stabilise_A="scale"
    # )
    # data = data.standardised_data

    # # Get posterior means for training
    # ts_means = get_posterier_train_state_means(data, trainer)
    
    # # Define the decoder:
    # latent_dim = data.train_states[0].shape[-1]
    # network = recognition.MLP([10, 10, 10])
    # decoder = DecoderNetwork(
    #     network=network,
    #     output_dim=latent_dim
    # )
    # decoder_trainer = DecoderTrainer(
    #     model=decoder,
    #     batch_size=64,
    #     iterations=2000
    # )

    # # Train the decoder 
    # decoder_trainer.fit(
    #     x=ts_means,  # NxTxD (Num sequences, Timesteps, Dimension)
    #     y=data.train_states,  # NxTxK  (k: latent dimension)
    #     use_pbar=True,
    # )

    # return data, trainer, free_energy, rp_ssm, ts_means, decoder, decoder_trainer
    

    # if __name__ == main():
    # ...
    # # Plot the predictions
    # plot_coordinate_walk_decoder_prediction(
    #     data=data,
    #     trainer=trainer,
    #     decoder_trainer=decoder_trainer,
    #     fig_shape=(2, 3),
    #     include_linreg=True,
    #     file_name="coordinate_decoder_walk_predictions",
    #     plot_dir="experiments/coordinate/plots",
    # )
    # plot_log_loss(
    #     loss=decoder_trainer.loss_tot,
    #     file_name="coordinate_decoder_log_loss",
    #     title="Decoder Log Loss",
    #     plot_dir="experiments/coordinate/plots",
    # )

