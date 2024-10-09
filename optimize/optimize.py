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

class BaseOptimizer:

    def __init__(self, args):
        self.args = args
        #self.args.compile = True
        self.robot = args.robot
        self.config = load_config(self.robot)[f'{args.chain}']

        try:
            os.makedirs("optuna")
        except OSError:
            pass

        try:
            os.makedirs(f"optuna/{self.robot}")
        except OSError:
            pass

        self.train_data = self.config["train_data"]
        self.test_data = self.config["test_data"]
        self.train_dataset, self.test_dataset = self.load_data(self.train_data, self.test_data, self.robot, self.config)
        self.args.cuda = True
        self.args.decay_epochs = 0
        self.args.fk_lr = None

    @abstractmethod
    def train_once(self, trial):
        pass

    @abstractmethod
    def optimize(self):
        pass

    def load_data(self, train_data, test_data, robot, config):
        train_dataset = IKDataset(root=train_data, robot=robot, config=config, mode="train")
        test_dataset = IKDataset(root=test_data, robot=robot, config=config, mode="test")
        return train_dataset, test_dataset