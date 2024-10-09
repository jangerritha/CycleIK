import argparse
import os
import random
from numpy import array, array_equal, allclose
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from cycleik_pytorch import IKDataset, GenericGenerator, GenericDiscriminator
from cycleik_pytorch import load_config, renormalize_pose, normalize_pose, slice_fk_pose, renormalize_joint_state
import pytorch_kinematics as pk
import time
import matplotlib.pyplot as plt
import pandas as ps
from abc import abstractmethod
from train.train_fk import FKTrainer
import optuna
from .optimize import BaseOptimizer
import joblib


class FKOptimizer(BaseOptimizer):

    def train_once(self, trial):
        torch.set_num_threads(2)

        args = self.args

        args.batch_size = trial.suggest_int('batch_size', 100, 1000, step=50)
        # epochs = args.epochs
        args.lr = None
        args.fk_lr = trial.suggest_float('lr', 0.00001, 0.001, step=0.00001)
        # if robot_dof <= 6:
        #    nbr_layers = trial.suggest_int('nbr_layers', 5, 10)
        # else:
        nbr_layers = trial.suggest_int('nbr_layers', 7, 9)
        layers = []
        for hidden_layer in range(nbr_layers):
            # layer = None
            if hidden_layer < nbr_layers / 2:
                layer = trial.suggest_int('layer{0}_neurons'.format(hidden_layer), 500, 3500, step=10)
            else:
                layer = trial.suggest_int('layer{0}_neurons'.format(hidden_layer), 10, 2000, step=10)
            layers.append(layer)
        args.layers = layers
        args.nbr_tanh = trial.suggest_int('nbr_tanh', 1, 3)
        # activation = trial.suggest_int('activation', 0, 3)
        args.activation = "GELU"  # trial.suggest_categorical("activation", ["GELU", "LeakyReLu"])#, "SELU", "CELU"])

        print(
            "Hyperparams - batch_size: {0}, lr: {1}, nbr_layers: {4}, nbr_tanh: {2}, activation: {3}\n neurons: {5}".format(
                args.batch_size, args.fk_lr, args.nbr_tanh, args.activation, nbr_layers, layers))

        trainer = FKTrainer(args, trial=trial, config=self.config, train_dataset=self.train_dataset, test_dataset=self.test_dataset)

        return trainer.train()

    def optimize(self):
        study = None
        if self.args.db is not None:
            # You can change the number of GPUs per trial here:
            study = optuna.create_study(study_name="cycleik_fk_optimizer",
                                        direction='minimize',
                                        pruner=optuna.pruners.HyperbandPruner(),
                                        sampler=optuna.samplers.TPESampler(),
                                        storage=f'sqlite:///{self.args.db}',
                                        load_if_exists=True)
        else:
            # You can change the number of GPUs per trial here:
            study = optuna.create_study(study_name="cycleik_fk_optimizer",
                                        direction='minimize',
                                        pruner=optuna.pruners.HyperbandPruner(),
                                        sampler=optuna.samplers.TPESampler())

        if self.args.add_seed:
            initial_layers = self.config["FKNet"]["architecture"]["layers"]
            initial_nbr_layers = len(initial_layers)

            initial_params = {"batch_size": self.config["FKNet"]["training"]["batch_size"],
                              "lr": self.config["FKNet"]["training"]["lr"],
                              "nbr_layers": initial_nbr_layers,
                              "nbr_tanh": self.config["FKNet"]["architecture"]["nbr_tanh"], }
            # "activation": config["IKNet"]["architecture"]["activation"]}

            for i in range(initial_nbr_layers):
                initial_params[f"layer{i}_neurons"] = initial_layers[i]

            study.enqueue_trial(initial_params)

        study.optimize(self.train_once, n_trials=self.args.trials)

        joblib.dump(study, f"./optuna/{self.robot}/cycleik_fk_optimizer.pkl")
        print("Best Config:\n {0}".format(study.best_params))