# Copyright (c) 2023. Jan-Gerrit Habekost. GNU General Public License https://www.gnu.org/licenses/gpl-3.0.html.

import argparse
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
import rospy
import bio_ik_msgs.srv

np.set_printoptions(precision=32)

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
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")

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

rospy.init_node("bio_ik_experiment")

rospy.wait_for_service("/moveit/bio_ik/get_bio_ik")
get_bio_ik = rospy.ServiceProxy("/moveit/bio_ik/get_bio_ik", bio_ik_msgs.srv.GetIK)

def calc_dist_to_home(x):
    home = np.array([0.613, -0.51315, 1.1728, 0., 0., 0.7075, 0.7075])

    r_target = R.from_quat(list(x[3:]))
    r_result = R.from_quat(list(home[3:]))
    # print(r_result)
    rotation_diff = r_target * r_result.inv()
    rotation_diff_save = rotation_diff.as_euler('zyx', degrees=True)
    rotation_diff = np.abs(rotation_diff_save)

    return np.concatenate((
        np.abs(np.subtract(home[:3], x[:3])), rotation_diff
    ))

no_solution_counter = 0

for i, data in progress_bar:#for i, data in progress_bar
    # get batch size data
    real_B = data["real_B"]

    real_b_np = real_B.clone().numpy()

    request = bio_ik_msgs.msg.IKRequest()
    request.group_name = "r_arm"
    request.timeout.nsecs = 10000000
    request.approximate = True
    request.avoid_collisions = True

    #request.ik_link_names.append("r_laser")
    request.pose_goals.append(bio_ik_msgs.msg.PoseGoal())
    request.pose_goals[-1].link_name = "r_laser"
    request.pose_goals[-1].weight = 2000
    request.pose_goals[-1].rotation_scale = 0.5
    request.pose_goals[-1].pose.position.x = real_b_np[0, 0]
    request.pose_goals[-1].pose.position.y = real_b_np[0, 1]
    request.pose_goals[-1].pose.position.z = real_b_np[0, 2]
    request.pose_goals[-1].pose.orientation.x = real_b_np[0, 3]
    request.pose_goals[-1].pose.orientation.y = real_b_np[0, 4]
    request.pose_goals[-1].pose.orientation.z = real_b_np[0, 5]
    request.pose_goals[-1].pose.orientation.w = real_b_np[0, 6]

    request.avoid_joint_limits_goals.append(bio_ik_msgs.msg.AvoidJointLimitsGoal())
    request.avoid_joint_limits_goals[-1].weight = 500
    request.avoid_joint_limits_goals[-1].primary = False

    request.minimal_displacement_goals.append(bio_ik_msgs.msg.MinimalDisplacementGoal())
    request.minimal_displacement_goals[-1].weight = 500
    request.minimal_displacement_goals[-1].primary = False

    response = get_bio_ik(request).ik_response

    ik_error = None
    if response.error_code.val != 1:
        no_solution_counter += 1
        ik_error = calc_dist_to_home(np.reshape(np.copy(real_b_np), newshape=(7)))
    else:
        fk_tensor = chain.forward_kinematics(torch.Tensor(np.array(response.solution.joint_state.position[26:34])).to(device))
        forward_result = slice_fk_pose(fk_tensor, batch_size=1)


        forward_result_np = forward_result.clone().detach().cpu().numpy()

        r_target = R.from_quat(list(real_b_np[0,3:]))
        r_result = R.from_quat(list(forward_result_np[0,3:]))
        #print(r_result)
        rotation_diff = r_target * r_result.inv()
        rotation_diff_save = rotation_diff.as_euler('zyx', degrees=True)
        rotation_diff = np.abs(rotation_diff_save)

        ik_error = np.concatenate(
            (np.reshape(np.abs(forward_result_np[:, :3] - real_b_np[:, :3]), newshape=(3)),
            np.array(rotation_diff)), axis=0)

    if not (ik_error[0] == 0. and ik_error[1] == 0. and ik_error[2] == 0.) and not (ik_error[3] == 0. and ik_error[4] == 0. and ik_error[5] == 0.):
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

    rospy.sleep(0.00001)

samples = len(error_list)

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

for k, row in enumerate(error_array_mean):
    if row[1] == 0.:
        print("xxx")

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

print(f'Nbr Failed Requests: {no_solution_counter}\n')

print("--------- IK Results -------------")
#print("Average IK Error: ", avg_ik_error[0,:3])
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
#print("Average IK Error: " + str(ik_error))

bins = [0.0]
for i in range(100):
    bins.append(bins[-1] + 0.4)

rot_bins = [0.0]
for i in range(100):
    rot_bins.append(rot_bins[-1] + 0.1)

fig, ax = plt.subplots(2)

ax[0].set_xlabel('Error (mm)')
ax[0].set_ylabel('Count')
n, bins, patches = ax[0].hist(np.multiply(error_array_mean[:, :1], 1000), bins=bins)
ax[0].axvline(np.mean(np.multiply(error_array_mean[:, :1], 1000), axis=0), color='k', linestyle='dashed', linewidth=1)
ax[0].axvline(np.mean(np.multiply(error_array_mean_cleaned[:, :1], 1000), axis=0), color='green', linestyle='dashed', linewidth=1)
bins = bins + 0.2
ax[0].plot(bins[:100], n,linewidth=1, linestyle=(0, (5, 1)), color="black")
ax[0].set_xticks(np.arange(0.0, 42.5, 2.5))
ax[0].set_yticks(np.arange(0.0, 26000., 1000.))

ax[1].set_xlabel('Error (degree)')
ax[1].set_ylabel('Count')
n, bins, patches = ax[1].hist(error_array_mean[:, 1:], bins=rot_bins)
ax[1].axvline(error_array_mean[:, 1:].mean(), color='k', linestyle='dashed', linewidth=1)
ax[1].axvline(error_array_mean_cleaned[:, 1:].mean(), color='green', linestyle='dashed', linewidth=1)
bins = bins + 0.05
ax[1].plot(bins[:100], n,linewidth=1, linestyle=(0, (5, 1)), color="black")
ax[1].set_xticks(np.arange(0.0, 11.0, 1.0))
ax[1].set_yticks(np.arange(0.0, 26000., 1000.))

fig.tight_layout()
plt.show()