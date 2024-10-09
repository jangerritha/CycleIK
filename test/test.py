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


class BaseTester:

    def __init__(self, args):
        torch.cuda.empty_cache()

        if args.manualSeed is None:
            self.manualSeed = random.randint(1, 10000)
        else:
            self.manualSeed = args.manualSeed
        print("Random Seed: ", self.manualSeed)
        random.seed(self.manualSeed)
        torch.manual_seed(self.manualSeed)

        cudnn.benchmark = True

        if torch.cuda.is_available() and not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        self.finetune = args.finetune
        self.robot = args.robot
        self.network = args.network
        self.config = load_config(self.robot)[f'{args.chain}']
        self.autoencoder = args.autoencoder
        self.two_stage = args.two_stage
        self.val_data = self.config["val_data"]
        self.robot_dof = self.config["robot_dof"]
        self.robot_urdf = self.config["robot_urdf"]
        self.robot_eef = self.config["robot_eef"]
        self.core_model_config = load_config(args.core_model)
        self.core_model = args.core_model

        self.use_ga = args.use_ga
        self.use_optimizer = args.use_optimizer

        # Dataset
        self.load_dataset()

        self.device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")

        self.model = None
        self.fk_model = None
        self.create_model()

        # Load state dicts
        if not self.finetune:
            self.model.load_state_dict(
            torch.load(os.path.join("weights", str(args.robot), f"model_{args.network}_with_kinematics.pth"), map_location=self.device))

        # Set model mode
        self.model.eval()
        if self.autoencoder:
            self.fk_model.eval()

        self.progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))


        self.samples = self.dataset.get_size()



        # rospy.init_node('ik_data_sampler', anonymous=True)

        """  setup  """

        # marker_publisher = rospy.Publisher("/moveit/visualization_marker", Marker, queue_size=100000)

        chain = pk.build_serial_chain_from_urdf(open(self.robot_urdf).read(), self.robot_eef)
        self.chain = chain.to(dtype=torch.float32, device=self.device)
        single_renormalize_move, single_renormalize, workspace_renormalize_move, workspace_renormalize = self.dataset.get_norm_params()

        self.single_renormalize_move = torch.Tensor(single_renormalize_move).to(self.device)
        self.single_renormalize = torch.Tensor(single_renormalize).to(self.device)
        self.workspace_renormalize_move = torch.Tensor(workspace_renormalize_move).to(self.device)
        self.workspace_renormalize = torch.Tensor(workspace_renormalize).to(self.device)

        self.count_bullshit = 0
        self.count_success = 0

    @abstractmethod
    def test_step(self, **kwargs):
        pass

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def process_test_losses(self, **kwargs):
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def test(self):
        pass