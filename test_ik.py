# Copyright (c) 2023. Jan-Gerrit Habekost. GNU General Public License https://www.gnu.org/licenses/gpl-3.0.html.

import argparse
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from cycleik_pytorch import Generator, IKDataset, Discriminator, GenericGenerator
from cycleik_pytorch import load_config, renormalize_pose, normalize_pose, slice_fk_pose, renormalize_joint_state
import pytorch_kinematics as pk
import matplotlib.pyplot as plt

np.set_printoptions(precision=32)

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

torch.cuda.empty_cache()

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
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False, pin_memory=True)

device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")

# create model
model = GenericGenerator(input_size=7, output_size=robot_dof,
                            nbr_tanh=config["IKNet"]["architecture"]["nbr_tanh"],
                            activation=config["IKNet"]["architecture"]["activation"],
                            layers=config["IKNet"]["architecture"]["layers"]).to(device)

# Load state dicts
model.load_state_dict(torch.load(os.path.join("weights", str(args.robot), "netG_B2A_with_kinematics.pth"), map_location=device))

# Set model mode
model.eval()

progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

error_list = []

samples = dataset.get_size()

min_error = 10000
max_error = 0
avg_error = 0

min_error_ik = 10000
max_error_ik = 0
avg_ik_error = [0., 0.,0.,0.,0.,0.]

over_1mm = 0
over_2_5mm = 0
over_5mm = 0
over_1cm = 0
over_1_5cm = 0
over_2cm = 0
over_5cm = 0
over_10cm = 0

over_1mm_ik = 0
over_2_5mm_ik = 0
over_5mm_ik = 0
over_1cm_ik = 0
over_1_5cm_ik = 0
over_2cm_ik = 0
over_5cm_ik = 0
over_10cm_ik = 0

it_counter = 0

chain = pk.build_serial_chain_from_urdf(open(robot_urdf).read(), robot_eef)
chain = chain.to(dtype=torch.float32, device=device)
single_renormalize_move, single_renormalize, workspace_renormalize_move, workspace_renormalize = dataset.get_norm_params()

single_renormalize_move = torch.Tensor(single_renormalize_move).to(device)
single_renormalize = torch.Tensor(single_renormalize).to(device)
workspace_renormalize_move = torch.Tensor(workspace_renormalize_move).to(device)
workspace_renormalize = torch.Tensor(workspace_renormalize).to(device)

count_failed_tests = 0
count_success = 0
for i, data in progress_bar:#for i, data in progress_bar
    # get batch size data
    gt_A = data["gt_A"].to(device)
    gt_B = data["gt_B"].to(device)
    real_B = data["real_B"].to(device)

    with torch.no_grad():
        result_B = model(gt_B)

    js = renormalize_joint_state(result_B, batch_size=len(gt_B), single_renormalize_move=single_renormalize_move, single_renormalize=single_renormalize)
    fk_tensor = chain.forward_kinematics(js)
    forward_result = slice_fk_pose(fk_tensor, batch_size=len(gt_B))

    gt_b_np = gt_B.detach().cpu().numpy()
    real_b_np = real_B.detach().cpu().numpy()
    forward_result_np = forward_result.clone().detach().cpu().numpy()

    r_target = R.from_quat(list(gt_b_np[:, 3:]))
    r_result = R.from_quat(list(forward_result_np[:, 3:]))
    rotation_diff = r_target * r_result.inv()
    rotation_diff_save = rotation_diff.as_euler('zyx', degrees=True)
    rotation_diff = np.abs(rotation_diff_save)

    ik_error_batch = np.concatenate(
        (np.reshape(np.abs(forward_result_np[:, :3] - real_b_np[:, :3]), newshape=(10000, 3)),
        np.array(rotation_diff)), axis=1)

    for k, ik_error in enumerate(ik_error_batch):
        if np.sum(ik_error[:3]) / 3 < 0.01 and np.sum(ik_error[3:]) / 3 < 20.:
            count_success += 1
        if ik_error[0] == 0. and ik_error[1] == 0. and ik_error[2] == 0.:
            print(k)
            print("--------------------")

        error_list.append(ik_error)
        avg_ik_error = np.add(avg_ik_error, ik_error)
        ik_error = np.average(ik_error[:3])

        if min_error_ik > ik_error:
            min_error_ik = ik_error
        if max_error_ik < ik_error:
            max_error_ik = ik_error


        if ik_error > 0.001:
            over_1mm_ik += 1
        if ik_error > 0.0025:
            over_2_5mm_ik += 1
        if ik_error > 0.005:
            over_5mm_ik += 1
        if ik_error > 0.01:
            over_1cm_ik += 1
        if ik_error > 0.015:
            over_1_5cm_ik += 1
        if ik_error > 0.02:
            over_2cm_ik += 1
        if ik_error > 0.05:
            over_5cm_ik += 1
        if ik_error > 0.1:
            over_10cm_ik += 1

avg_ik_error = np.true_divide(avg_ik_error, samples)
avg_ik_error_mm = np.copy(avg_ik_error)
avg_ik_error_mm[:3] = np.multiply(avg_ik_error_mm[:3], 1000)

error_array = np.array(error_list, dtype=np.float32)

error_array_mean = np.concatenate((
    np.reshape(np.average(error_array[:, :3], axis=1), newshape=(samples, 1)),
    np.reshape(np.average(error_array[:, 3:], axis=1), newshape=(samples, 1))
), axis=1)

error_array_mean_cleaned = np.copy(error_array_mean)

outlier_rows = []
for r, row in enumerate(error_array_mean):
    if row[0] > 0.01 or row[1] > 1.:
        outlier_rows.append(r)

error_array_mean_cleaned = np.delete(error_array_mean_cleaned, outlier_rows, axis=0)

print(len(error_array_mean_cleaned))

max_pos_error_all_axis, max_rot_error_all_axis = np.max(error_array[:,:3], axis=0), np.max(error_array[:,3:], axis=0)
min_pos_error_all_axis, min_rot_error_all_axis = np.min(error_array[:,:3], axis=0), np.min(error_array[:,3:], axis=0)
max_pos_err_all_axis_mm = np.multiply(max_pos_error_all_axis, 1000)
min_pos_err_all_axis_mm = np.multiply(min_pos_error_all_axis, 1000)

max_pos_err, max_rot_error = np.max(error_array_mean, axis=0)[0], np.max(error_array_mean, axis=0)[1]
min_pos_err, min_rot_error = np.min(error_array_mean, axis=0)[0], np.min(error_array_mean, axis=0)[1]
max_pos_err_mm = np.multiply(max_pos_err, 1000)
min_pos_err_mm = np.multiply(min_pos_err, 1000)
mean_pos_err = np.average(avg_ik_error_mm[:3])
mean_rot_err = np.average(avg_ik_error_mm[3:])

print(f'\nAverage IK error over all axis: XYZ (mm): {avg_ik_error_mm[:3]}, RPY: {avg_ik_error_mm[3:]}')
print(f'Max IK error over all axis: XYZ (mm): {max_pos_err_all_axis_mm}, RPY: {max_rot_error_all_axis}')
print(f'Min IK error over all axis: XYZ (mm): {min_pos_err_all_axis_mm}, RPY: {min_rot_error_all_axis}\n')

print(f'\nAverage IK error over all axis: XYZ (mm): {mean_pos_err}, RPY: {mean_rot_err}')
print(f'Max IK error : XYZ (mm): {max_pos_err_mm}, RPY: {max_rot_error}')
print(f'Min IK error: XYZ (mm): {min_pos_err_mm}, RPY: {min_rot_error}\n')

print(f"Success Rate: {count_success / samples}")

print("--------- IK Results -------------")
print("Average IK Error: \n"
      "     Position (mm): {0}".format(avg_ik_error[:3]) + "\n"
      "     Orientation (degree): {0}".format(avg_ik_error[3:]))
print("Min Error: " + str(min_error_ik))
print("Max Error: " + str(max_error_ik))
print("Error > 1mm: ", over_1mm_ik)
print("Percent > 1mm: ", over_1mm_ik / samples)
print("Error > 2.5mm: ", over_2_5mm_ik)
print("Percent > 2.5mm: ", over_2_5mm_ik / samples)
print("Error > 5mm: ", over_5mm_ik)
print("Percent > 5mm: ", over_5mm_ik / samples)
print("Error > 1cm: ", over_1cm_ik)
print("Percent > 1cm: ", over_1cm_ik / samples)
print("Error > 1.5cm: ", over_1_5cm_ik)
print("Percent > 1.5cm: ", over_1_5cm_ik / samples)
print("Error > 2cm: ", over_2cm_ik)
print("Percent > 2cm: ", over_2cm_ik / samples)
print("Error > 5cm: ", over_5cm_ik)
print("Percent > 5cm: ", over_5cm_ik / samples)
print("Error > 10cm: ", over_10cm_ik)
print("Percent > 10cm: ", over_10cm_ik / samples)

bins = [0.0]
for i in range(240):
    bins.append(bins[-1] + 0.005)

rot_bins = [0.0]
for i in range(320):
    rot_bins.append(rot_bins[-1] + 0.0005)

fig, ax = plt.subplots(2)

ax[0].set_xlabel('Error (mm)')
ax[0].set_ylabel('Count')
n, bins, patches = ax[0].hist(np.multiply(error_array_mean[:, :1], 1000), bins=bins)
mean_error = ax[0].axvline(np.mean(np.multiply(error_array_mean[:, :1], 1000), axis=0), color='orange', linestyle='dashed', linewidth=1)
mean_error.set_label('Mean Absolute Error (all solutions)')
mean_error =ax[0].axvline(np.mean(np.multiply(error_array_mean_cleaned[:, :1], 1000), axis=0), color='green', linestyle='dashed', linewidth=1)
mean_error.set_label('Mean Absolute Error (only valid solutions)')
bins = bins + 0.0025
ax[0].plot(bins[:240], n,linewidth=1, linestyle=(0, (5, 1)), color="black")
ax[0].set_xticks(np.arange(0.0, 1.4, 0.2))
ax[0].set_yticks(np.arange(0.0, 7000., 1000.))
ax[0].xaxis.label.set_fontweight('bold')
ax[0].yaxis.label.set_fontweight('bold')
ax[0].legend()

ax[1].set_xlabel('Error (degree)')
ax[1].set_ylabel('Count')
n, bins, patches = ax[1].hist(error_array_mean[:, 1:], bins=rot_bins)
mean_error = ax[1].axvline(error_array_mean[:, 1:].mean(), color='orange', linestyle='dashed', linewidth=1)
mean_error.set_label('Mean Absolute Error (all solutions)')
mean_error = ax[1].axvline(error_array_mean_cleaned[:, 1:].mean(), color='green', linestyle='dashed', linewidth=1)
mean_error.set_label('Mean Absolute Error (only valid solutions)')
bins = bins + 0.00025
ax[1].plot(bins[:320], n,linewidth=1, linestyle=(0, (5, 1)), color="black")
ax[1].set_xticks(np.arange(0.0, 0.18, 0.02))
#ax[1].set_xlim(right=0.16)
ax[1].set_yticks(np.arange(0.0, 5000., 1000.))
ax[1].xaxis.label.set_fontweight('bold')
ax[1].yaxis.label.set_fontweight('bold')
ax[1].legend()

fig.tight_layout()
plt.savefig('./img/losses/vis_error_dist_iknet.png')
plt.show()