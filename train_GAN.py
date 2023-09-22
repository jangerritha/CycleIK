# Copyright (c) 2023. Jan-Gerrit Habekost. GNU General Public License https://www.gnu.org/licenses/gpl-3.0.html.

import torch.utils.data
from cycleik_pytorch import DecayLR, IKDataset, Generator, Discriminator, GenericNoisyGenerator, NoisyGenerator
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
import pytorch_kinematics as pk

train_dataset, test_dataset = None, None

def load_data(train_data, test_data, robot, config):
    global train_dataset, test_dataset
    train_dataset = IKDataset(root=train_data, robot=robot, config=config)
    test_dataset = IKDataset(root=test_data, robot=robot, config=config)

def train_gan(args, trial=None, config=None):
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

    if args.lr is None:
        args.lr = config["GAN"]["training"]["lr"]
    if args.batch_size is None:
        args.batch_size = config["GAN"]["training"]["batch_size"]
    if args.noise_vector_size is None:
        args.noise_vector_size = config["GAN"]["architecture"]["noise_vector_size"]
    noise_vector_size = args.noise_vector_size
    print(args)

    device_name = f"cuda:{args.gpu}" if args.cuda else "cpu"
    device = torch.device(device_name)

    # Dataset
    if not optimizer_run:
        load_data(train_data, test_data, args.robot, config)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=True, pin_memory=True, num_workers=1)
    train_data_size, test_data_size = train_dataset.get_size(), test_dataset.get_size()

    torch.autograd.set_detect_anomaly(True)

    # create model

    if optimizer_run:
        model = GenericNoisyGenerator(input_size=7, output_size=robot_dof,
                                    noise_vector_size=noise_vector_size,
                                    nbr_tanh=args.nbr_tanh,
                                    activation=args.activation,
                                    layers=args.layers).to(device)
    else:

        model = GenericNoisyGenerator(input_size=7, output_size=robot_dof,
                                         noise_vector_size=noise_vector_size,
                                         nbr_tanh=config["GAN"]["architecture"]["nbr_tanh"],
                                         activation=config["GAN"]["architecture"]["activation"],
                                         layers=config["GAN"]["architecture"]["layers"]).to(device)

    if args.model != "" and args.model is not None:
        model.load_state_dict(torch.load(args.model))

    cycle_loss = torch.nn.L1Loss().to(device)
    variance_loss = torch.nn.MSELoss().to(device)

    # Optimizer
    optimizer_G = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    lr_lambda = DecayLR(args.epochs, 0, args.decay_epochs).step
    lr_scheduler_G_linear = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda, verbose=True)

    z_zero_mean = torch.zeros(args.batch_size, noise_vector_size, dtype=torch.float32).to(device)

    js_samples = 2000
    z_zero_mean_js_samples = torch.zeros(js_samples, noise_vector_size, dtype=torch.float32).to(device)
    one_tensor_js_samples = torch.ones(8, dtype=torch.float32).to(device)
    js_samples_tensor = torch.zeros(js_samples, 7, dtype=torch.float32).to(device)

    if train_data_size % args.batch_size != 0:
        small_z_zero_mean = torch.zeros(train_data_size % args.batch_size, noise_vector_size, dtype=torch.float32).to(device)
        small_js_samples_tensor = torch.zeros(train_data_size % js_samples, 7, dtype=torch.float32).to(device)
        small_z_zero_mean_js_samples = torch.zeros(train_data_size % js_samples, noise_vector_size, dtype=torch.float32).to(device)

    train_losses = []
    var_losses = []
    val_losses = []
    val_var_losses = []

    chain = pk.build_serial_chain_from_urdf(open(robot_urdf).read(), robot_eef)
    chain = chain.to(dtype=torch.float32, device=device)
    single_renormalize_move, single_renormalize, workspace_move, workspace_renormalize = train_dataset.get_norm_params()
    single_renormalize_move = torch.Tensor(single_renormalize_move).to(device)
    single_renormalize = torch.Tensor(single_renormalize).to(device)
    workspace_move = torch.Tensor(workspace_move).to(device)
    workspace_renormalize = torch.Tensor(workspace_renormalize).to(device)

    for epoch in range(0, args.epochs):
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        loss_fk_avg = 0
        loss_cycle_avg = 0

        train_loss = 0
        zero_loss = 0
        val_loss = 0
        var_loss = 0

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

            z = None
            if train_data_size % args.batch_size != 0 and i == len(train_dataloader) - 1:
                z = ((torch.rand(train_data_size % args.batch_size, noise_vector_size) - 0.)).to(device)
            else:
                z = ((torch.rand(args.batch_size, noise_vector_size) - 0.)).to(device)

            backward_B2A = model(z, gt_B)

            bs = args.batch_size
            if len(gt_B) != args.batch_size: bs = len(gt_B)
            js = renormalize_joint_state(backward_B2A, bs, single_renormalize=single_renormalize,
                                         single_renormalize_move=single_renormalize_move)

            fk_tensor = chain.forward_kinematics(js)

            forward_result = slice_fk_pose(fk_tensor, bs)

            cycle_loss_B2A = cycle_loss(forward_result, real_B)

            loss_for_lr = cycle_loss_B2A.clone()
            avg_loss += loss_for_lr * (len(gt_B) / args.batch_size)

            if train_data_size % args.batch_size != 0 and i == len(train_dataloader) - 1:
                index = torch.randint(low=0, high=train_data_size % args.batch_size, size=(1, 1))
            else:
                index = torch.randint(low=0, high=args.batch_size, size=(1, 1))

            # print(gt_B[index].shape)
            reshape_index = torch.reshape(gt_B[index], shape=(1, 7))

            js_samples_tensor = reshape_index.repeat(js_samples, 1)
            z = (torch.rand(js_samples, noise_vector_size)/1.).to(device)
            backward_B2A = model(z, js_samples_tensor)

            variance_loss_B2A = variance_loss(torch.mean(torch.var(input=backward_B2A, dim=0)),
                                              torch.mean(torch.var(input=z, dim=0)))

            errG = variance_loss_B2A * 6. + cycle_loss_B2A * 1.

            errG.backward()
            optimizer_G.step()

            if epoch == 0 or (epoch + 1) % 1 == 0:
                train_loss += cycle_loss_B2A.item() * (len(gt_B) / args.batch_size)
                var_loss += variance_loss_B2A.item() * (len(gt_B) / args.batch_size)

            progress_bar.set_description(
                f"[{epoch}/{args.epochs - 1}][{i}/{len(train_dataloader) - 1}] "
                # f"zero_loss:: {zero_joints_loss_ik.item():.8f} "
                f"cycle_loss:: {cycle_loss_B2A.item():.8f} "
                f"variance_loss:: {variance_loss_B2A.item():.8f} "
                #    f"cycle_loss_var: {cycle_loss_var.item():.8f} "
            )

        if epoch % 5 == 0 and not optimizer_run:
            torch.save(model.state_dict(), f"weights/{args.robot}/netG_B2A_GAN_epoch_{epoch}_with_kinematics.pth")

        val_var_loss = 0.
        val_loss = 0.

        if epoch == 0 or (epoch + 1) % 10 == 0:
            val_z_zero_mean = torch.zeros(10000, noise_vector_size, dtype=torch.float32).to(device)
            val_z_zero_mean_small = torch.zeros(test_data_size % 10000, noise_vector_size, dtype=torch.float32).to(device)
            val_z_zero_mean_js_samples = torch.zeros(js_samples, noise_vector_size, dtype=torch.float32).to(device)
            # one_tensor_js_samples = torch.ones(8, dtype=torch.float32).to(device)
            val_js_samples_tensor = torch.zeros(js_samples, 7, dtype=torch.float32).to(device)

            progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
            for i, data in progress_bar:
                # get batch size data
                gt_A = data["gt_A"].to(device)
                gt_B = data["gt_B"].to(device)
                real_B = data["real_B"].to(device)

                with torch.no_grad():
                    bs = len(gt_B)

                    z = ((torch.rand(bs, noise_vector_size) - 0.)).to(device)
                    backward_B2A = model(z, gt_B)


                    js = renormalize_joint_state(backward_B2A, bs, single_renormalize=single_renormalize,
                                                 single_renormalize_move=single_renormalize_move)

                    fk_tensor = chain.forward_kinematics(js)
                    forward_result = slice_fk_pose(fk_tensor, bs)

                    cycle_loss_B2A_val = cycle_loss(forward_result, real_B)

                    index = torch.randint(low=0, high=bs, size=(1, 1))

                    reshape_index = torch.reshape(gt_B[index], shape=(1, 7))
                    val_js_samples_tensor = reshape_index.repeat(js_samples, 1)
                    z = val_z_zero_mean_js_samples + 1 * ((torch.rand(js_samples, noise_vector_size) - 0.) ).to(device)
                    backward_B2A = model(z, val_js_samples_tensor)

                    variance_loss_B2A = variance_loss(torch.mean(torch.var(input=backward_B2A, dim=0)),
                                                      torch.mean(torch.var(input=z, dim=0)))
                    val_var_loss += variance_loss_B2A.cpu().numpy() * (len(gt_B) / 10000)
                    val_loss += cycle_loss_B2A_val.cpu().numpy() * (len(gt_B) / 10000)

                # Set G_A and G_B's gradients to zero
                progress_bar.set_description(
                    f"[{epoch}/{args.epochs - 1}][{i}/{len(test_dataloader) - 1}] "
                    f"cycle_loss:: {cycle_loss_B2A_val.item():.8f} "
                    #f"variance_loss:: {variance_loss_B2A.item():.8f} "
                )

            val_losses.append(val_loss/ (test_data_size / 10000))
            val_var_losses.append(val_var_loss/ (test_data_size / 10000))

        train_losses.append(train_loss / (train_data_size / args.batch_size))
        var_losses.append(var_loss / (train_data_size / args.batch_size))
        avg_loss = avg_loss / (train_data_size / args.batch_size)
        lr_scheduler_G_linear.step()

        if optimizer_run:
            trial.report(avg_loss, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        print("Avg Loss: {0}".format(avg_loss))

    print([train_losses, var_losses, val_losses, val_var_losses])

    if not optimizer_run:
        # save last check pointing
        torch.save(model.state_dict(), f"weights/{args.robot}/netG_B2A_GAN_with_kinematics.pth")
        with open(rf"results/{args.robot}/train_GAN_with_kinematics_loss.p", "wb") as output_file:
            pickle.dump([train_losses, var_losses, val_losses, val_var_losses], output_file)

    return val_losses[-1] + (val_var_losses[-1] * 10)


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
    parser.add_argument("--noise_vector_size", type=int, default=None,
                        help="learning rate. (default:0.0002)")
    parser.add_argument("--noise", type=float, default=None,
                        help="noise on position for fk. (default:0.0002)")
    parser.add_argument("-p", "--print-freq", default=100, type=int,
                        metavar="N", help="print frequency. (default:100)")
    parser.add_argument("--cuda", action="store_true", help="Enables cuda")
    parser.add_argument("--model", default="", help="path to netG_B2A (to continue training)")
    parser.add_argument("--manualSeed", type=int, help="Seed for initializing training. (default:none)")
    parser.add_argument("--robot", type=str, default="nicol", help="Robot model IK is trained for")
    parser.add_argument("--gpu", type=str, default="0", help="GPU used for training")
    train_args = parser.parse_args()

    train_gan(train_args)
