# Copyright (c) 2023. Jan-Gerrit Habekost. GNU General Public License https://www.gnu.org/licenses/gpl-3.0.html.

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
from cycleik_pytorch import Generator, IKDataset, Discriminator, NoisyGenerator, GenericNoisyGenerator
from cycleik_pytorch import load_config, renormalize_pose, normalize_pose, slice_fk_pose, renormalize_joint_state

import pytorch_kinematics as pk
import time
import matplotlib.pyplot as plt

# Set the axes title font size
plt.rc('axes', labelsize=14)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=12)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=12)
# Set the legend font size
plt.rc('legend', fontsize=9)
# Set the font size of the figure title
plt.rc('figure', titlesize=12)

parser = argparse.ArgumentParser(
    description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")
parser.add_argument("--robot", type=str, default="nicol", help="Robot model IK is trained for")
parser.add_argument("--gpu", type=int, default=0, help="Robot model IK is trained for")
args = parser.parse_args()
print(args)

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

config = load_config(args.robot)

val_data = config["val_data"]
robot_dof = config["robot_dof"]
robot_urdf = config["robot_urdf"]
robot_eef = config["robot_eef"]

# Dataset
dataset = IKDataset(root=val_data, test=True, robot=args.robot, config=config)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")

noise_vector_size = config["GAN"]["architecture"]["noise_vector_size"]
js_samples = 1000

# create model
model = GenericNoisyGenerator(input_size=7, output_size=robot_dof,
                            noise_vector_size=noise_vector_size,
                            nbr_tanh=config["GAN"]["architecture"]["nbr_tanh"],
                            activation=config["GAN"]["architecture"]["activation"],
                            layers=config["GAN"]["architecture"]["layers"]).to(device)

# Load state dicts
model.load_state_dict(torch.load(os.path.join("weights", str(args.robot), "netG_B2A_GAN_with_kinematics.pth"), map_location=device))

# Set model mode
model.eval()

progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

fk_error = [0, 0, 0]
ik_error = [0, 0, 0]

samples = dataset.get_size()

avg_pos_error = 0.
min_pos_error = 0.
max_pos_error = 0.
avg_rot_error = 0.
min_rot_error = 0.
max_rot_error = 0.

avg_pos_error_all_axis = 0.
min_pos_error_all_axis = 0.
max_pos_error_all_axis = 0.
avg_rot_error_all_axis = 0.
min_rot_error_all_axis = 0.
max_rot_error_all_axis = 0.

it_counter = 0

avg_pos_error_only_valid = 0
avg_rot_error_only_valid = 0
validdd_counter = 0

chain = pk.build_serial_chain_from_urdf(open(robot_urdf).read(), robot_eef)
chain = chain.to(dtype=torch.float32, device=device)
single_renormalize_move, single_renormalize, workspace_renormalize_move, workspace_renormalize = dataset.get_norm_params()

single_renormalize_move = torch.Tensor(single_renormalize_move).to(device)
single_renormalize = torch.Tensor(single_renormalize).to(device)
workspace_renormalize_move = torch.Tensor(workspace_renormalize_move).to(device)
workspace_renormalize = torch.Tensor(workspace_renormalize).to(device)

count_failed_tests = 0

z_zero_mean = torch.zeros(js_samples, noise_vector_size, dtype=torch.float32).to(device)
ik_pose_batch = torch.zeros(js_samples, 7, dtype=torch.float32).to(device)

measure_time = 0.

avg_pos_error_list = []
avg_rot_error_list = []

count_success = 0

for i, data in progress_bar:#for i, data in progress_bar
    # get batch size data
    gt_A = data["gt_A"].to(device)
    gt_B = data["gt_B"].to(device)
    real_B = data["real_B"].to(device)

    #pose = normalize_pose()
    with torch.no_grad():
        start_time = time.time()
        z = z_zero_mean + 1 * torch.rand(js_samples, noise_vector_size).to(device)
        result_B = model(z, gt_B.repeat((js_samples, 1)))
        end_time = time.time()
        measure_time += end_time - start_time

    js = renormalize_joint_state(result_B, batch_size=js_samples, single_renormalize_move=single_renormalize_move,
                                 single_renormalize=single_renormalize)
    fk_tensor = chain.forward_kinematics(js)
    forward_result = slice_fk_pose(fk_tensor, batch_size=js_samples)

    gt_b_np = gt_B.detach().cpu().numpy()
    real_b_np = real_B.repeat((js_samples, 1)).detach().cpu().numpy()
    forward_result_np = forward_result.clone().detach().cpu().numpy()

    r_target = R.from_quat(list(real_b_np[:, 3:]))
    r_result = R.from_quat(list(forward_result_np[:, 3:]))
    rotation_diff = r_target * r_result.inv()
    rotation_diff_save = rotation_diff.as_euler('zyx', degrees=True)
    rotation_diff = np.abs(rotation_diff_save)

    ik_error_batch = np.concatenate(
        (np.reshape(np.abs(forward_result_np[:, :3] - real_b_np[:, :3]), newshape=(js_samples, 3)),
         np.array(rotation_diff)), axis=1)

    for error in ik_error_batch:
        if np.sum(error[:3]) / 3 < 0.01 and np.sum(error[3:]) / 3 < 20.0:
            count_success += 1 / js_samples

    ik_error_batch[:, :3] = np.multiply(ik_error_batch[:, :3], 1000)

    for error_batch in ik_error_batch:
        temp_mean_pos_err = np.average(error_batch[:3], axis=0)
        temp_mean_rot_err = np.average(error_batch[3:], axis=0)
        if temp_mean_pos_err <= 10.0 and temp_mean_rot_err <= 20.0:
            avg_pos_error_only_valid += temp_mean_pos_err
            avg_rot_error_only_valid += temp_mean_rot_err
            validdd_counter += 1

    max_pos_err_all_axes = np.max(ik_error_batch[:, :3], axis=0)
    min_pos_err_all_axes = np.min(ik_error_batch[:, :3], axis=0)
    avg_pos_err_all_axes = np.average(ik_error_batch[:, :3], axis=0)

    max_rot_err_all_axes = np.max(ik_error_batch[:, 3:], axis=0)
    min_rot_err_all_axes = np.min(ik_error_batch[:, 3:], axis=0)
    avg_rot_err_all_axes = np.average(ik_error_batch[:, 3:], axis=0)

    max_pos_err = np.max(np.mean(ik_error_batch[:, :3], axis=1))
    min_pos_err = np.min(np.mean(ik_error_batch[:, :3], axis=1))
    avg_pos_err = np.mean(np.mean(ik_error_batch[:,:3], axis=0))

    max_rot_err = np.max(np.mean(ik_error_batch[:, 3:], axis=1))
    min_rot_err = np.min(np.mean(ik_error_batch[:, 3:], axis=1))
    avg_rot_err = np.mean(np.mean(ik_error_batch[:, 3:], axis=0))

    avg_pos_error_list.append(avg_pos_err)
    avg_rot_error_list.append(avg_rot_err)
    avg_pos_error += avg_pos_err / samples
    min_pos_error += min_pos_err / samples
    max_pos_error += max_pos_err / samples
    avg_rot_error += avg_rot_err / samples
    min_rot_error += min_rot_err / samples
    max_rot_error += max_rot_err / samples

    avg_pos_error_all_axis += avg_pos_err_all_axes / samples
    min_pos_error_all_axis += min_pos_err_all_axes / samples
    max_pos_error_all_axis += max_pos_err_all_axes / samples
    avg_rot_error_all_axis += avg_rot_err_all_axes / samples
    min_rot_error_all_axis += min_rot_err_all_axes / samples
    max_rot_error_all_axis += max_rot_err_all_axes / samples


avg_pos_error_only_valid = avg_pos_error_only_valid / validdd_counter
avg_rot_error_only_valid = avg_rot_error_only_valid / validdd_counter
print("--------- IK Results -------------")
print(f"\nAverage position Error: {avg_pos_error}")
print(f"Average min position Error: {min_pos_error}")
print(f"Average max position Error: {max_pos_error}\n")

print(f"\nAverage orientation Error: {avg_rot_error}")
print(f"Average min orientation Error: {min_rot_error}")
print(f"Average max orientation Error: {max_rot_error}\n")

print(f"\nAverage position Error all axis: {avg_pos_error_all_axis}")
print(f"Average min position Error all axis: {min_pos_error_all_axis}")
print(f"Average max position Error all axis: {max_pos_error_all_axis}\n")

print(f"\nAverage orientation Error all axis: {avg_rot_error_all_axis}")
print(f"Average min orientation Error all axis: {min_rot_error_all_axis}")
print(f"Average max orientation Error all axis: {max_rot_error_all_axis}\n")

print(f'Mean inference time: {measure_time / samples}')
print(f'Success: {count_success / samples}')

avg_pos_error_array = np.array(avg_pos_error_list)
avg_rot_error_array = np.array(avg_rot_error_list)

bins = [0.0]
for i in range(100):
    bins.append(bins[-1] + 0.1)

rot_bins = [0.0]
for i in range(100):
    rot_bins.append(rot_bins[-1] + 0.01)

fig, ax = plt.subplots(2)

ax[0].set_xlabel('Error (mm)')
ax[0].set_ylabel('Count')
n, bins, patches = ax[0].hist(avg_pos_error_array, bins=bins)
mean_error = ax[0].axvline(np.mean(avg_pos_error_array, axis=0), color='orange', linestyle='dashed', linewidth=1)
mean_error.set_label('Mean Absolute Error')
mean_error = ax[0].axvline(avg_pos_error_only_valid, color='green', linestyle='dashed', linewidth=1)
mean_error.set_label('Mean Absolute Error (only valid solutions)')
#ax[0].axvline(np.mean(np.multiply(error_array_mean_cleaned[:, :1], 1000), axis=0), color='green', linestyle='dashed', linewidth=1)
bins = bins + 0.05
ax[0].plot(bins[:100], n,linewidth=1, linestyle=(0, (5, 1)), color="black")
ax[0].set_xticks(np.arange(0.0, 11.0, 1.0))
ax[0].set_yticks(np.arange(0.0, 22500., 4500.))
ax[0].xaxis.label.set_fontweight('bold')
ax[0].yaxis.label.set_fontweight('bold')
ax[0].legend()

ax[1].set_xlabel('Error (degree)')
ax[1].set_ylabel('Count')
n, bins, patches = ax[1].hist(avg_rot_error_array, bins=rot_bins)
mean_error = ax[1].axvline(avg_rot_error_array.mean(), color='orange', linestyle='dashed', linewidth=1)
mean_error.set_label('Mean Absolute Error')
mean_error = ax[1].axvline(avg_rot_error_only_valid, color='green', linestyle='dashed', linewidth=1)
mean_error.set_label('Mean Absolute Error (only valid solutions)')
#ax[1].axvline(error_array_mean_cleaned[:, 1:].mean(), color='green', linestyle='dashed', linewidth=1)
bins = bins + 0.005
ax[1].plot(bins[:100], n,linewidth=1, linestyle=(0, (5, 1)), color="black")
ax[1].set_xticks(np.arange(0.0, 1.1, 0.1))
ax[1].set_yticks(np.arange(0.0, 15000., 2500.))
ax[1].xaxis.label.set_fontweight('bold')
ax[1].yaxis.label.set_fontweight('bold')
ax[1].legend()

fig.tight_layout()
plt.savefig('./img/losses/vis_error_dist_GAN.png')
plt.show()