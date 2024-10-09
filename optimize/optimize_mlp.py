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
from train import MLPTrainer
import optuna
from .optimize import BaseOptimizer
import joblib


class MLPOptimizer(BaseOptimizer):

    def train_once(self, trial):
        torch.set_num_threads(2)

        args = self.args

        args.batch_size = trial.suggest_int('batch_size', 100, 600, step=50)
        # epochs = args.epochs
        args.lr = trial.suggest_float('lr', 0.00001, 0.001, step=0.00001)
        # if robot_dof <= 6:
        #    nbr_layers = trial.suggest_int('nbr_layers', 5, 10)
        # else:
        nbr_layers = trial.suggest_int('nbr_layers', 5, 9)
        layers = []
        for hidden_layer in range(nbr_layers):
            # layer = None
            layer = trial.suggest_int('layer{0}_neurons'.format(hidden_layer), 500, 3500, step=10)
            #if hidden_layer < nbr_layers / 2:
            #    layer = trial.suggest_int('layer{0}_neurons'.format(hidden_layer), 500, 3500, step=10)
            #else:
            #    layer = trial.suggest_int('layer{0}_neurons'.format(hidden_layer), 10, 2000, step=10)
            layers.append(layer)
        args.layers = layers
        args.nbr_tanh = trial.suggest_int('nbr_tanh', 0, 3)
        # activation = trial.suggest_int('activation', 0, 3)
        args.activation = "GELU"  # trial.suggest_categorical("activation", ["GELU", "LeakyReLu"])#, "SELU", "CELU"])

        args.position_weight = trial.suggest_float('position_weight', 1., 20., step=1.)
        args.orientation_weight = trial.suggest_float('orientation_weight', 1., 20., step=1.)

        print("Hyperparams - batch_size: {0}, lr: {1}, nbr_layers: {4}, nbr_tanh: {2}, activation: {3}\n neurons: {5}".format(
                args.batch_size, args.lr, args.nbr_tanh, args.activation, nbr_layers, layers))

        trainer = MLPTrainer(args, trial=trial, config=self.config, train_dataset=self.train_dataset, test_dataset=self.test_dataset)

        return trainer.train()

    def optimize(self):
        study = None
        if self.args.study_name != "":
            study_name = self.args.study_name
        else:
            study_name = f"cycleik_ik_optimizer_{self.args.robot}_robot_gpu_{self.args.gpu}"
        if self.args.db is not None:
            # You can change the number of GPUs per trial here:
            study = optuna.create_study(study_name=study_name,
                                        direction='minimize',
                                        pruner=optuna.pruners.HyperbandPruner(),
                                        sampler=optuna.samplers.TPESampler(),
                                        storage=f'sqlite:///{self.args.db}',
                                        load_if_exists=True)
        else:
            # You can change the number of GPUs per trial here:
            study = optuna.create_study(study_name=study_name,
                                        direction='minimize',
                                        pruner=optuna.pruners.HyperbandPruner(),
                                        sampler=optuna.samplers.TPESampler())

        if self.args.add_seed:
            initial_layers = self.config["IKNet"]["architecture"]["layers"]
            initial_nbr_layers = len(initial_layers)

            initial_params = {"batch_size": self.config["IKNet"]["training"]["batch_size"],
                              "lr": self.config["IKNet"]["training"]["lr"],
                              "nbr_layers": initial_nbr_layers,
                              "nbr_tanh": self.config["IKNet"]["architecture"]["nbr_tanh"],
                              "position_weight": self.config["IKNet"]["position_weight"],
                              "orientation_weight": self.config["IKNet"]["orientation_weight"]}
            # "activation": config["IKNet"]["architecture"]["activation"]}

            for i in range(initial_nbr_layers):
                initial_params[f"layer{i}_neurons"] = initial_layers[i]

            study.enqueue_trial(initial_params)

            #if self.robot != "nicol":
            #    initial_layers = [2200, 2400, 2400, 1900, 250, 220, 30, 380]
            #    initial_nbr_layers = len(initial_layers)
            #
            #    initial_params = {"batch_size": 300,
            #                      "lr": 0.0001,
            #                      "nbr_layers": initial_nbr_layers,
            #                      "nbr_tanh": 3, }
            #    # "activation": config["IKNet"]["architecture"]["activation"]}
            #
            #    for i in range(initial_nbr_layers):
            #        initial_params[f"layer{i}_neurons"] = initial_layers[i]
            #
            #    study.enqueue_trial(initial_params)

        study.optimize(self.train_once, n_trials=self.args.trials)

        joblib.dump(study, f"./optuna/{self.robot}/cycleik_ik_optimizer_{self.args.gpu}.pkl")
        print("Best Config:\n {0}".format(study.best_params))