import torch.utils.data
import os
import random
import pickle
import torch.backends.cudnn as cudnn
import torch.utils.data
from tqdm import tqdm
import optuna
from cycleik_pytorch import DecayLR, IKDataset, GenericGenerator, GenericDiscriminator
from cycleik_pytorch import weights_init, load_config, slice_fk_pose, normalize_pose, renormalize_pose, renormalize_joint_state, JSD
import pytorch_kinematics as pk
from abc import abstractmethod
#from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt
from torch.functional import F
import math
import scipy.linalg as linalg

jsd = JSD()
sqrt_2 = None

class BaseTrainer:

    def __init__(self, args, trial=None, config=None, train_dataset=None, test_dataset=None):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        try:
            os.makedirs("weights")
        except OSError:
            pass

        try:
            os.makedirs("img/losses")
        except OSError:
            pass

        try:
            os.makedirs("img")
        except OSError:
            pass

        try:
            os.makedirs("results")
        except OSError:
            pass

        try:
            os.makedirs(f"weights/{args.robot}")
        except OSError:
            pass

        try:
            os.makedirs(f"results/{args.robot}")
        except OSError:
            pass

        self.optimizer_run = True if trial is not None else False
        torch.set_num_threads(2)

        try:
            if args.manualSeed is None:
                self.manualSeed = random.randint(1, 10000)
            else:
                self.manualSeed = args.manualSeed
        except AttributeError:
            self.manualSeed = random.randint(1, 10000)

        if self.optimizer_run:
            self.optimizer_run = True
            self.cuda = True
            self.decay_epochs = 0
            self.noise = None
            self.her = False
            self.trial = trial
            self.nbr_tanh = args.nbr_tanh
            self.activation = args.activation
            self.layers = args.layers

        print("Random Seed: ", self.manualSeed)
        random.seed(self.manualSeed)
        torch.manual_seed(self.manualSeed)

        cudnn.benchmark = True

        if torch.cuda.is_available() and not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        if config is None:
            self.config = load_config(args.robot)[f'{args.chain}']
        else:
            self.config = config

        self.autoencoder = args.autoencoder
        self.two_stage = args.two_stage

        self.train_data = self.config["train_data"]
        self.test_data = self.config["test_data"]
        self.robot_dof = self.config["robot_dof"]
        self.robot_urdf = self.config["robot_urdf"]
        self.robot_eef = self.config["robot_eef"]
        self.robot = args.robot
        if not self.optimizer_run:
            self.core_model = args.core_model
            self.core_model_config = load_config(args.core_model)[f'{args.core_model_chain}']

        self.js_samples = 100
        self.manifold_sample_nbr = 100
        self.epochs = args.epochs
        self.decay_epochs = args.decay_epochs

        self.network = None
        if args.network == "GAN":
            self.network = "GAN"
        elif args.network == "MLP":
            self.network = "IKNet"
        elif args.network == "FK":
            self.network = "FKNet"

        if self.autoencoder or self.network == "FKNet":
            if args.fk_lr is None:
                self.fk_lr = self.config["FKNet"]["training"]["lr"]
            else:
                self.fk_lr = args.fk_lr

        if args.lr is None:
            self.lr = self.config[self.network]["training"]["lr"]
        else:
            self.lr = args.lr
        if args.batch_size is None:
            self.batch_size = self.config[self.network]["training"]["batch_size"]
        else:
            self.batch_size = args.batch_size
        if self.network == "GAN":
            if args.noise_vector_size is None:
                self.noise_vector_size = self.config["GAN"]["architecture"]["noise_vector_size"]
            else:
                self.noise_vector_size = args.noise_vector_size
        print(args)

        device_name = f"cuda:{args.gpu}" if args.cuda else "cpu"
        self.device = torch.device(device_name)

        # Dataset
        if not self.optimizer_run:
            self.train_dataset, self.test_dataset = self.load_data(self.train_data, self.test_data, self.robot, self.config)
        else:
            self.train_dataset, self.test_dataset = train_dataset, test_dataset

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=1)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=10000, shuffle=True, pin_memory=True, num_workers=1)
        self.train_data_size, self.test_data_size = self.train_dataset.get_size(), self.test_dataset.get_size()

        torch.autograd.set_detect_anomaly(True)

        try:
            chain = pk.build_serial_chain_from_urdf(open(self.robot_urdf).read(), self.robot_eef, self.config['base_link'])
            self.chain = chain.to(dtype=torch.float32, device=self.device)
        except ValueError:
            chain = pk.build_serial_chain_from_urdf(open(self.robot_urdf).read(), self.robot_eef)
            self.chain = chain.to(dtype=torch.float32, device=self.device)
        single_renormalize_move, single_renormalize, workspace_move, workspace_renormalize = self.train_dataset.get_norm_params()
        self.single_renormalize_move = torch.Tensor(single_renormalize_move).to(self.device)
        self.single_renormalize = torch.Tensor(single_renormalize).to(self.device)
        self.workspace_move = torch.Tensor(workspace_move).to(self.device)
        self.workspace_renormalize = torch.Tensor(workspace_renormalize).to(self.device)


        self.fk_model = None
        self.model = None
        self.fk_optimizer = None
        self.optimizer = None
        self.fk_lr_scheduler = None
        self.lr_scheduler = None
        self.create_model()
        self.create_optimizer()
        self.create_lr_scheduler()
        self.progress_bar = None

        self.cycle_loss = torch.nn.SmoothL1Loss(beta=0.01)#torch.nn.L1Loss().to(self.device)
        self.position_cycle_loss = torch.nn.SmoothL1Loss(beta=0.001)#torch.nn.L1Loss().to(self.device)
        self.rotation_cycle_loss = torch.nn.SmoothL1Loss(beta=0.01, reduction='none')#torch.nn.L1Loss().to(self.device)
        self.kl_divergence_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)

        self.compile = args.compile
        self.adaptive_loss_order = 10
        self.gpu = args.gpu
        #self.softadapt = NormalizedSoftAdapt(beta=0.1, accuracy_order=self.adaptive_loss_order)

        sqrt_tensor  = torch.Tensor(1)
        sqrt_tensor[0] = 2
        global sqrt_2
        sqrt_2 = torch.sqrt(sqrt_tensor).to(self.device)

    def load_data(self, train_data, test_data, robot, config):
        train_dataset = IKDataset(root=train_data, robot=robot, config=config, mode="train")
        test_dataset = IKDataset(root=test_data, robot=robot, config=config, mode="test")
        return [train_dataset, test_dataset]

    @abstractmethod
    def training_step(self, **kwargs):
        pass

    @abstractmethod
    def validation_step(self, **kwargs):
        pass

    @abstractmethod
    def process_training_losses(self, **kwargs):
        pass

    @abstractmethod
    def process_validation_losses(self, **kwargs):
        pass

    def create_optimizer(self):
        if self.autoencoder or self.network == "FKNet":
            self.fk_optimizer = torch.optim.Adam(self.fk_model.parameters(), lr=self.fk_lr, betas=(0.9, 0.999))

    def create_lr_scheduler(self):
        if self.autoencoder or self.network == "FKNet":
            lr_lambda = DecayLR(self.epochs, 0, self.decay_epochs).step
            self.fk_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.fk_optimizer, lr_lambda=lr_lambda, verbose=True)

    def create_model(self):
        if self.autoencoder or self.network == "FKNet":
            if self.optimizer_run:
                self.fk_model = GenericDiscriminator(input_size=self.robot_dof, output_size=7,
                                                  nbr_tanh=self.nbr_tanh,
                                                  activation=self.activation,
                                                  layers=self.layers).to(self.device)
            else:
                self.fk_model = GenericDiscriminator(input_size=self.robot_dof, output_size=7,
                                                  nbr_tanh=self.config["FKNet"]["architecture"]["nbr_tanh"],
                                                  activation=self.config["FKNet"]["architecture"]["activation"],
                                                  layers=self.config["FKNet"]["architecture"]["layers"]).to(self.device)
            if self.two_stage:
                self.fk_model.load_state_dict(torch.load(f'./weights/{self.robot}/model_FK_with_kinematics.pth'))

    @abstractmethod
    def create_checkpoint(self, epoch):
        pass

    @abstractmethod
    def train(self):
        pass

    @staticmethod
    def orientation_loss_phi_2(quat_1, quat_2):
        cycle_loss_B2A_orientation_positive = torch.sum(torch.pow(torch.subtract(quat_1, quat_2), 2), dim=1)
        cycle_loss_B2A_orientation_negative = torch.sum(torch.pow(torch.add(quat_1, quat_2), 2), dim=1)
        cycle_loss_orientation = torch.minimum(cycle_loss_B2A_orientation_positive, cycle_loss_B2A_orientation_negative)
        return torch.mean(cycle_loss_orientation, dim=0)

    @staticmethod
    def orientation_loss_phi_4(quat_1, quat_2):
        cycle_loss_orientation = torch.subtract(1, torch.abs(torch.sum(torch.multiply(quat_1, quat_2), dim=1)))
        return torch.mean(cycle_loss_orientation, dim=0)

    @staticmethod
    def orientation_loss_phi_7(quat_1, quat_2):
        cycle_loss_B2A_orientation_positive = torch.sum(torch.nn.functional.smooth_l1_loss(quat_1, quat_2, beta=0.01, reduction='none'), dim=1)
        cycle_loss_B2A_orientation_negative = torch.sum(torch.nn.functional.smooth_l1_loss(-quat_1, quat_2, beta=0.01, reduction='none'), dim=1)
        cycle_loss_orientation = torch.minimum(cycle_loss_B2A_orientation_positive, cycle_loss_B2A_orientation_negative)
        return torch.mean(cycle_loss_orientation, dim=0)

    @staticmethod
    def kl_divergence(predicted_distribution, target_distribution, beta=0.1):
        joint_space_distribution = torch.nn.functional.log_softmax(predicted_distribution, dim=1)
        target_distribution = torch.nn.functional.log_softmax(target_distribution, dim=1)

        #point_wise_kl_div = self.kl_divergence_loss(joint_space_distribution, target_distribution)
        point_wise_kl_div = torch.nn.functional.kl_div(joint_space_distribution, target_distribution, reduction="none", log_target=True)
        row_wise_kl_div = torch.sum(point_wise_kl_div, dim=1)
        row_wise_kl_div = torch.subtract(row_wise_kl_div, beta)
        row_wise_kl_div = torch.maximum(row_wise_kl_div, torch.zeros(row_wise_kl_div.size()).to(f'cuda:{row_wise_kl_div.get_device()}'))
        #smooth_kl_div = row_wise_kl_div.where(row_wise_kl_div > beta,
        #                                      torch.multiply(torch.square(row_wise_kl_div), 0.1))
        #smooth_kl_div = torch.nn.functional.smooth_l1_loss()
        #smooth_kl_div = point_wise_kl_div.where(condition=point_wise_kl_div < beta,
        #                                        input=torch.multiply(torch.square(point_wise_kl_div), beta))
        return torch.mean(row_wise_kl_div, dim=0)
        #return torch.mean(smooth_kl_div, dim=0)

    @staticmethod
    def min_distance_js_loss(predicted_distribution, target_distribution, beta=0.1):
        point_wise_diff = torch.abs(torch.subtract(predicted_distribution, target_distribution))
        #print(point_wise_diff.size())
        #print(point_wise_diff)
        point_wise_diff = point_wise_diff.where(
            (point_wise_diff > beta),
            torch.multiply(torch.square(point_wise_diff), beta)
        )
        return torch.mean(torch.mean(point_wise_diff, dim=1), dim=0)

    @staticmethod
    def batchmean_kl_div(predicted_distribution, target_distribution):
        joint_space_distribution = torch.nn.functional.log_softmax(predicted_distribution, dim=1)
        target_distribution = torch.nn.functional.log_softmax(target_distribution, dim=1)

        # point_wise_kl_div = self.kl_divergence_loss(joint_space_distribution, target_distribution)
        point_wise_kl_div = torch.nn.functional.kl_div(joint_space_distribution, target_distribution, reduction="batchmean",
                                                       log_target=True)
        #row_wise_kl_div = torch.sum(point_wise_kl_div, dim=1)
        #smooth_kl_div = row_wise_kl_div.where(row_wise_kl_div < beta,
        #                                      torch.multiply(torch.square(row_wise_kl_div), 1))
        # smooth_kl_div = torch.nn.functional.smooth_l1_loss()
        # smooth_kl_div = point_wise_kl_div.where(condition=point_wise_kl_div < beta,
        #                                        input=torch.multiply(torch.square(point_wise_kl_div), beta))
        return point_wise_kl_div#torch.mean(smooth_kl_div, dim=0)



    @staticmethod
    def max_variance_loss(predicted_distribution):
        return torch.subtract(torch.div(4, 12), torch.mean(torch.var(predicted_distribution, dim=0)))
