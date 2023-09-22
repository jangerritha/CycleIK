# Copyright (c) 2023. Jan-Gerrit Habekost. GNU General Public License https://www.gnu.org/licenses/gpl-3.0.html.

import torch.utils.data
import argparse
import os
import torch.utils.data
from cycleik_pytorch import load_config
import optuna
import joblib

parser = argparse.ArgumentParser(
    description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
parser.add_argument("--robot", type=str, default="nicol", help="Robot model IK is trained for")
parser.add_argument("--trials", default=100, type=int, metavar="N", help="number of total trials in optimizer")
parser.add_argument("--epochs", default=10, type=int, metavar="N", help="number of total trials in optimizer")
parser.add_argument("--gpu", type=str, default="0", help="GPU used for training")
args = parser.parse_args()

from train_GAN import train_gan, load_data

config = load_config(args.robot)
train_data = config["train_data"]
test_data = config["test_data"]

def train(trial):
    torch.set_num_threads(2)

    try:
        os.makedirs("optuna")
    except OSError:
        pass

    try:
        os.makedirs(f"optuna/{args.robot}")
    except OSError:
        pass

    args.batch_size = trial.suggest_int('batch_size', 100, 600, step=50)
    #epochs = args.epochs
    args.lr = trial.suggest_float('lr', 0.00001, 0.001, step=0.00001)
    #if robot_dof <= 6:
    #    nbr_layers = trial.suggest_int('nbr_layers', 5, 10)
    #else:
    nbr_layers = trial.suggest_int('nbr_layers', 7, 9)
    layers = []
    for hidden_layer in range(nbr_layers):
        #layer = None
        if hidden_layer < nbr_layers / 2:
            layer = trial.suggest_int('layer{0}_neurons'.format(hidden_layer), 500, 3500, step=10)
        else:
            layer = trial.suggest_int('layer{0}_neurons'.format(hidden_layer), 10, 2000, step=10)
        layers.append(layer)
    args.layers = layers
    args.nbr_tanh = trial.suggest_int('nbr_tanh', 1, 3)
    #activation = trial.suggest_int('activation', 0, 3)
    args.noise_vector_size = trial.suggest_int('noise_vector_size', 3, 10, step=1)
    args.activation = "GELU"#trial.suggest_categorical("activation", ["GELU", "LeakyReLu"])#, "SELU", "CELU"])

    print("Hyperparams - batch_size: {0}, lr: {1}, nbr_layers: {4}, nbr_tanh: {2}, activation: {3}\n neurons: {5}".format(args.batch_size, args.lr, args.nbr_tanh, args.activation, nbr_layers, layers))

    return train_gan(args, trial, config)

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    study = optuna.create_study(study_name="cycleik_gan_optimizer",
                                direction='minimize',
                                pruner=optuna.pruners.HyperbandPruner(),
                                sampler=optuna.samplers.TPESampler())

    initial_layers = config["GAN"]["architecture"]["layers"]
    initial_nbr_layers = len(initial_layers)

    initial_params ={"batch_size": config["GAN"]["training"]["batch_size"],
     "lr": config["GAN"]["training"]["lr"],
     "nbr_layers": initial_nbr_layers,
     "nbr_tanh": config["GAN"]["architecture"]["nbr_tanh"],
     "noise_vector_size": config["GAN"]["architecture"]["noise_vector_size"]}

    for i in range(initial_nbr_layers):
        initial_params[f"layer{i}_neurons"] = initial_layers[i]

    study.enqueue_trial(initial_params)

    load_data(train_data, test_data, args.robot, config)

    study.optimize(train, n_trials=args.trials)

    joblib.dump(study, f"./optuna/{args.robot}/cycleik_GAN_optimizer.pkl")
    print("Best Config:\n {0}".format(study.best_params))
