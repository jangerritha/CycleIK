# Copyright (c) 2023. Jan-Gerrit Habekost. GNU General Public License https://www.gnu.org/licenses/gpl-3.0.html.

import argparse
import os
import random
import pickle
import torch.backends.cudnn as cudnn
import torch.utils.data
from tqdm import tqdm
import optuna
from cycleik_pytorch import DecayLR, IKDataset, Generator, Discriminator, GenericGenerator
from cycleik_pytorch import load_config, slice_fk_pose, normalize_pose, renormalize_pose, renormalize_joint_state
import numpy as np
import pytorch_kinematics as pk
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation as R

train_dataset, test_dataset = None, None

def load_data(train_data, test_data, robot, config):
    global train_dataset, test_dataset
    train_dataset = IKDataset(root=train_data, robot=robot, config=config)
    test_dataset = IKDataset(root=test_data, robot=robot, config=config)

def train_ik(args, trial=None, config=None):
    torch.cuda.empty_cache()

    try:
        os.makedirs("weights")
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

    optimizer_run = True if trial is not None else False
    torch.set_num_threads(2)

    try:
        if args.manualSeed is None:
            args.manualSeed = random.randint(1, 10000)
    except AttributeError:
        args.manualSeed = random.randint(1, 10000)

    if optimizer_run:
        args.cuda = True
        args.model = None
        args.decay_epochs = 0
        args.noise = None
        args.her = False

    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if config is None:
        config = load_config(args.robot)

    train_data = config["train_data"]
    test_data = config["test_data"]
    robot_dof = config["robot_dof"]
    robot_urdf = config["robot_urdf"]
    robot_eef = config["robot_eef"]

    try:
        robot_zero_joints = config["zero_joints_goal"]
    except KeyError:
        robot_zero_joints = None
        pass

    zero_joints_goal = True if robot_zero_joints is not None and len(robot_zero_joints) > 0 else False


    if args.lr is None:
        args.lr = config["IKNet"]["training"]["lr"]
    if args.batch_size is None:
        args.batch_size = config["IKNet"]["training"]["batch_size"]

    print(args)

    device_name = f"cuda:{args.gpu}" if args.cuda else "cpu"
    device = torch.device(device_name)

    # Dataset
    if not optimizer_run:
        load_data(train_data, test_data, args.robot, config)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=1)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=True, pin_memory=False, num_workers=1)
    train_data_size, test_data_size = train_dataset.get_size(), test_dataset.get_size()

    torch.autograd.set_detect_anomaly(True)
    # create model
    if optimizer_run:
        model = GenericGenerator(input_size=7, output_size=robot_dof,
                                    nbr_tanh=args.nbr_tanh,
                                    activation=args.activation,
                                    layers=args.layers).to(device)
    else:
        model = GenericGenerator(input_size=7, output_size=robot_dof,
                                    nbr_tanh=config["IKNet"]["architecture"]["nbr_tanh"],
                                    activation=config["IKNet"]["architecture"]["activation"],
                                    layers=config["IKNet"]["architecture"]["layers"]).to(device)

    if args.model != "" and args.model is not None:
        model.load_state_dict(torch.load(args.model))

    cycle_loss = torch.nn.L1Loss().to(device)
    zero_joints_loss = torch.nn.MSELoss().to(device)


    # Optimizers
    optimizer_G = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    lr_lambda = DecayLR(args.epochs, 0, args.decay_epochs).step
    lr_scheduler_G_linear = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda, verbose=True)


    g_losses = []
    d_losses = []

    identity_losses = []
    gan_losses = []
    cycle_losses = []

    zero_joints_array = None
    small_zero_joints_array = None
    if zero_joints_goal:
        zero_joints_array = torch.zeros((args.batch_size, len(robot_zero_joints))).to(device)
        if train_data_size % args.batch_size != 0:
            small_zero_joints_array = torch.zeros((train_data_size % args.batch_size, len(robot_zero_joints))).to(device)

    center_joints_array = np.zeros(shape=(args.batch_size, 8), dtype=np.float32)
    for r in range(args.batch_size):
        center_joints_array[r] = np.array([0.5, 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
    center_joints_array = torch.Tensor(center_joints_array).to(device)

    train_losses = []
    zero_losses = []
    val_losses = []

    chain = pk.build_serial_chain_from_urdf(open(robot_urdf).read(), robot_eef)
    chain = chain.to(dtype=torch.float32, device=device)
    single_renormalize_move, single_renormalize, workspace_move, workspace_renormalize = train_dataset.get_norm_params()
    single_renormalize_move = torch.Tensor(single_renormalize_move).to(device)
    single_renormalize = torch.Tensor(single_renormalize).to(device)
    workspace_move = torch.Tensor(workspace_move).to(device)
    workspace_renormalize = torch.Tensor(workspace_renormalize).to(device)

    def calculate_zero_joints_loss(joint_state, batch_size):
        zero_joints_loss_ik = None
        joint_index_view = None
        for joint_index in robot_zero_joints:
            if joint_index_view is None:
                joint_index_view = joint_state.select(dim=1, index=joint_index)
                joint_index_view = joint_index_view.reshape(shape=(len(joint_index_view), 1))
            else:
                joint_index_view = torch.concat((joint_index_view, joint_state.select(dim=1, index=joint_index).reshape(
                    shape=(len(joint_index_view), 1))), dim=-1)
        if batch_size != args.batch_size:
            zero_joints_loss_ik = zero_joints_loss(joint_index_view, small_zero_joints_array)
        else:
            zero_joints_loss_ik = zero_joints_loss(joint_index_view, zero_joints_array)
        return zero_joints_loss_ik

    for epoch in range(0, args.epochs):
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        loss_fk_avg = 0
        loss_cycle_avg = 0

        train_loss = 0
        zero_loss = 0
        val_loss = 0

        avg_fk_timeout = 0

        x = torch.zeros(args.batch_size, 7, dtype=torch.float32).to(device)

        avg_loss = 0.
        last_avg_loss = None
        for i, data in progress_bar:
            # get batch size data
            gt_A = data["gt_A"].to(device)
            gt_B = data["gt_B"].to(device)
            real_B = data["real_B"].to(device)

            optimizer_G.zero_grad()

            backward_B2A = model(gt_B)

            bs = len(gt_B) if len(gt_B) != args.batch_size else args.batch_size

            js = renormalize_joint_state(backward_B2A, bs, single_renormalize=single_renormalize, single_renormalize_move=single_renormalize_move)

            fk_tensor = chain.forward_kinematics(js)

            forward_result = slice_fk_pose(fk_tensor, bs)

            if args.noise is not None:
                if len(gt_B) != args.batch_size:
                    x = torch.zeros(bs, 7, dtype=torch.float32).to(device)
                x = x + (args.noise ** 0.5) * torch.randn(bs, 7).to(device)
                forward_result = forward_result + x

            cycle_loss_B2A = cycle_loss(forward_result, real_B)
            loss_for_lr = cycle_loss_B2A.clone()
            avg_loss += loss_for_lr * (len(gt_B) / args.batch_size)

            if zero_joints_goal:
                zero_joints_loss_ik = calculate_zero_joints_loss(backward_B2A, bs)
                errG = cycle_loss_B2A * 5000 + zero_joints_loss_ik
            else:
                errG = cycle_loss_B2A * 1000

            errG.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)

            # Update G_A and G_B's weights
            optimizer_G.step()

            if zero_joints_goal:
                if epoch == 0 or (epoch + 1) % 1 == 0:
                    train_loss += cycle_loss_B2A.item() * (len(gt_B) / args.batch_size)
                    zero_loss += zero_joints_loss_ik.item() * (len(gt_B) / args.batch_size)

                progress_bar.set_description(
                    f"[{epoch}/{args.epochs - 1}][{i}/{len(train_dataloader) - 1}] "
                    f"zero_loss:: {zero_joints_loss_ik.item():.8f} "
                    f"cycle_loss:: {cycle_loss_B2A.item():.8f} ")
            else:
                if epoch == 0 or (epoch + 1) % 1 == 0:
                    train_loss += cycle_loss_B2A.item() * (len(gt_B) / args.batch_size)

                progress_bar.set_description(
                    f"[{epoch}/{args.epochs - 1}][{i}/{len(train_dataloader) - 1}] "
                    f"cycle_loss:: {cycle_loss_B2A.item():.8f} ")

        # do check pointing
        if epoch % 5 == 0 and not optimizer_run:
            torch.save(model.state_dict(), f"weights/{args.robot}/netG_B2A_epoch_{epoch}_with_kinematics.pth")

        if epoch == 0 or (epoch + 1) % 10 == 0:
            progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
            for i, data in progress_bar:
                # get batch size data
                gt_A = data["gt_A"].to(device)
                gt_B = data["gt_B"].to(device)
                real_B = data["real_B"].to(device)

                with torch.no_grad():
                    backward_B2A = model(gt_B)

                    bs = len(gt_B) if len(gt_B) != args.batch_size else args.batch_size

                    js = renormalize_joint_state(backward_B2A, bs, single_renormalize=single_renormalize, single_renormalize_move=single_renormalize_move)

                    fk_tensor = chain.forward_kinematics(js)

                    forward_result = slice_fk_pose(fk_tensor, bs)

                    cycle_loss_B2A_val = cycle_loss(forward_result, real_B)

                    val_loss += cycle_loss_B2A_val.item() * (len(gt_B) / 10000)

                # Set G_A and G_B's gradients to zero
                progress_bar.set_description(
                    f"[{epoch}/{args.epochs - 1}][{i}/{len(test_dataloader) - 1}] "
                    f"fk_loss:: {cycle_loss_B2A_val.item():.8f} ")

            val_losses.append(val_loss / (test_data_size / 10000))

        train_losses.append(train_loss / (train_data_size / args.batch_size))
        zero_losses.append(zero_loss / (train_data_size / args.batch_size))
        avg_loss = avg_loss / (train_data_size / args.batch_size)
        lr_scheduler_G_linear.step()

        if optimizer_run:
            trial.report(avg_loss, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        print("Avg Loss: {0}".format(avg_loss))

    print([train_losses, zero_losses, val_losses])

    if not optimizer_run:
        # save last check pointing
        torch.save(model.state_dict(), f"weights/{args.robot}/netG_B2A_with_kinematics_{args.gpu}.pth")
        with open(rf"./results/{args.robot}/train_ik_loss_with_kinematics_{args.gpu}.p", "wb") as output_file:
            pickle.dump([train_losses, zero_losses, val_losses], output_file)

    return val_losses[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
    parser.add_argument("--epochs", default=100, type=int, metavar="N",
                        help="number of total epochs to run")
    parser.add_argument("--decay_epochs", type=int, default=0,
                        help="epoch to start linearly decaying the learning rate to 0. (default:100)")
    parser.add_argument("-b", "--batch-size", default=None, type=int,
                        metavar="N",
                        help="mini-batch size (default: 1), this is the total "
                             "batch size of all GPUs on the current node when "
                             "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--lr", type=float, default=None,
                        help="learning rate. (default:0.0002)")
    parser.add_argument("--noise", type=float, default=None,
                        help="noise on position for fk. (default:0.0002)")
    parser.add_argument("--cuda", action="store_true", help="Enables cuda")
    parser.add_argument("--model", default="", help="path to netG_B2A (to continue training)")
    parser.add_argument("--manualSeed", type=int, help="Seed for initializing training. (default:none)")
    parser.add_argument("--robot", type=str, default="nicol", help="Robot model IK is trained for")
    parser.add_argument("--gpu", type=str, default="0", help="GPU used for training")
    train_args = parser.parse_args()

    train_ik(train_args)
