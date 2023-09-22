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
import time
import matplotlib.pyplot as plt

from gaikpy import robot_definitions

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
robot_urdf = './assets/urdf/NICOL.urdf' #config["robot_urdf"]
robot_eef = config["robot_eef"]

# Dataset
dataset = IKDataset(root=val_data, test=True, robot=args.robot, config=config)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")

# create model
model = GenericGenerator(input_size=7, output_size=robot_dof,
                            nbr_tanh=config["IKNet"]["architecture"]["nbr_tanh"],
                            activation=config["IKNet"]["architecture"]["activation"],
                            layers=config["IKNet"]["architecture"]["layers"]).to(device)

# Load state dicts
model.load_state_dict(torch.load("./weights/nicol/netG_B2A_with_kinematics.pth", map_location=device))

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

#rospy.init_node('ik_data_sampler', anonymous=True)

"""  setup  """
error_x = 0.
error_y = 0.
error_z = 0.



#marker_publisher = rospy.Publisher("/moveit/visualization_marker", Marker, queue_size=100000)


chain = pk.build_serial_chain_from_urdf(open(robot_urdf).read(), robot_eef)
chain = chain.to(dtype=torch.float32, device=device)
single_renormalize_move, single_renormalize, workspace_renormalize_move, workspace_renormalize = dataset.get_norm_params()

single_renormalize_move = torch.Tensor(single_renormalize_move).to(device)
single_renormalize = torch.Tensor(single_renormalize).to(device)
workspace_renormalize_move = torch.Tensor(workspace_renormalize_move).to(device)
workspace_renormalize = torch.Tensor(workspace_renormalize).to(device)

count_failed_tests = 0

visualisation = False
nicol = robot_definitions.NICOL_robot(visualisation=visualisation)

def transform_from_rot_trans(tf, rot):
    """Transforation matrix from rotation matrix and translation vector."""
    rot = rot.reshape(3, 3)
    tf = tf.reshape(3, 1)
    return np.vstack((np.hstack([rot, tf]), [0, 0, 0, 1]))

count_success = 0
time_measure = 0
for i, data in progress_bar:#for i, data in progress_bar
    # get batch size data
    gt_A = data["gt_A"].to(device)
    gt_B = data["gt_B"].to(device)
    real_B = data["real_B"].to(device)
    #print(real_images_B.cpu().detach().numpy().shape)
    #print("-----------------------------------------------")
    start_time = (time.time() * 1000)
    #print("----- IK test -------")
    with torch.no_grad():
        result_B = model(gt_B)

    #ik_error = np.average(abs(moveit_fk_array - gt_b_np), axis=0)

    js = renormalize_joint_state(result_B, batch_size=len(gt_B), single_renormalize_move=single_renormalize_move, single_renormalize=single_renormalize)
    js = js.repeat(50, 1).detach().cpu().numpy()
    end_time = (time.time() * 1000)

    time_measure += end_time - start_time
    #print(rotation_diff_save.shape)
    # print(abs(result_A_np[:,:3] - gt_b_np[:,:3]))
    # print(np.array(rotation_diff))


    #print(np.abs(np.multiply(forward_result[:, :3], single_normalize_pos_np) - np.multiply(gt_b_np[:, :3], single_normalize_pos_np)))
    #print(np.array(rotation_diff))

    #print(ik_error)
    #print("FK Moveit: ", moveit_fk_array)

    gt_b_np = gt_B.detach().cpu().numpy()
    real_b_np = real_B.detach().cpu().numpy()



    #print(error)
    #for k, joint_state in tqdm(enumerate(js), total=len(gt_B)):
    #print(joint_state)
    pose_matrix = transform_from_rot_trans(real_b_np[0, :3], R.from_quat(list(real_b_np[0, 3:])).as_matrix())
    tar_rot = R.from_quat(real_b_np[0, 3:])
    #tar_rot_matrix = tar_rot.as_matrix()
    #print(js)
    joint_positions = nicol.get_ik(pose_matrix, robot_chain="right_arm", dist_acc=0.001, or_acc=0.001,
                                   include_orientation=True, multiproc=True, num_generations=19, orientation_weight=0.15, nbr_seeds_per_proc=4,
                                   seed_individuals=js, mutation_rate=0.05, not_reachable_mode="warning")

    #fk_tensor = chain.forward_kinematics(torch.Tensor(joint_positions).to(device))
    #forward_result = slice_fk_pose(fk_tensor, batch_size=1)
    #forward_result_np = forward_result.clone().detach().cpu().numpy()

    fk_array = nicol.fast_fk("right_arm", joint_positions)
    # fk_tensor = chain.forward_kinematics(joint_positions)
    # forward_result = slice_fk_pose(fk_tensor, batch_size=1)
    # forward_result_np = forward_result.clone().detach().cpu().numpy()

    r_target = tar_rot  # R.from_quat(list(real_b_np[0, 3:]))
    r_result = R.from_matrix(fk_array[:3, :3])
    # print(r_result)
    rotation_diff = r_target * r_result.inv()
    rotation_diff_save = rotation_diff.as_euler('zyx', degrees=True)
    rotation_diff = np.abs(rotation_diff_save)

    linear_diff = np.abs(np.reshape(fk_array[:3, 3:], newshape=(3)) - real_b_np[0, :3])

    # print(f'rotation_diff: {rotation_diff}')
    # print(f'linear_diff: {linear_diff}')

    ik_error = np.concatenate(
        (linear_diff, np.array(rotation_diff)), axis=0)

    #r_result = R.from_quat(forward_result_np[0, 3:])
    ## print(r_result)
    #rotation_diff = r_target * r_result.inv()
    #rotation_diff_save = rotation_diff.as_euler('zyx', degrees=True)
    #rotation_diff = np.abs(rotation_diff_save)
    #ik_error = np.concatenate(
    #    (np.abs(np.reshape(forward_result_np[:, :3], newshape=(3)) - real_b_np[0, :3]),
    #     np.array(rotation_diff)))

    if np.sum(ik_error[:3]) / 3 < 0.01 and np.sum(ik_error[3:]) / 3 < 20.0:
        count_success += 1
    #    error_x += ik_error[0]
    #    error_y += ik_error[1]
    #    error_z += ik_error[2]
    #print(ik_error)

    #if ik_error[0] == 0. and ik_error[1] == 0. and ik_error[2] == 0.:
        #print(k)
    #    print("--------------------")
    #ik_error[:3] = np.multiply(ik_error[:3], 1000)
    #avg_ik_error = np.add(avg_ik_error, np.true_divide(ik_error, samples))
    #
    #ik_error[:3] = np.true_divide(ik_error[:3], 1000)
    #ik_error = np.average(ik_error[:3])

    #ik_error[:3] = np.multiply(ik_error[:3], 1000)
    #avg_ik_error = np.add(avg_ik_error, np.true_divide(ik_error, samples))

    error_list.append(ik_error)
    avg_ik_error = np.add(avg_ik_error, ik_error)

    #ik_error[:3] = np.true_divide(ik_error[:3], 1000)
    ik_error = np.average(ik_error[:3])

    #color_for_viz = None
    #if ik_error <= 0.001:
    #    color_for_viz = ColorRGBA(0.0, 255, 0.0, 0.8)
    #elif ik_error <= 0.01:
    #    color_for_viz = ColorRGBA(255, 255, 0.0, 0.8)
    #elif ik_error > 0.01:
    #    color_for_viz = ColorRGBA(255, 0.0, 0.0, 0.8)

    #if ik_error > 0.01:
    #    pos_marker = real_B[k, :3]
    #    if pos_marker[2] < 0.8:
    #        print('Suuuuus')
    #        count_failed_tests += 1
    #    ## print(pose)
    #    marker = Marker(
    #        type=Marker.SPHERE,
    #        action=Marker.ADD,
    #        id=(i * 10000 + k + 1),
    #        lifetime=rospy.Duration(),
    #        pose=Pose(Point(pos_marker[0], pos_marker[1], pos_marker[2]), Quaternion(0, 0, 0, 1)),
    #        scale=Vector3(0.001, 0.001, 0.001),
    #        header=Header(frame_id='world'),
    #        color=color_for_viz,
    #        # lifetime=0,
    #        frame_locked=False)
    #    marker_publisher.publish(marker)

    #if ik_error > 0.1:
    #    print(torch.mul(gt_B, single_normalize_pos))

    #print(ik_error)

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

    #rospy.sleep(0.00001)

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

print(f'success: {count_success/samples}')
print(f'inference time: {(time_measure/samples)+19}')

#print(f"Avg inference_time: {measure_time / samples}")

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

print(f"test errors: {error_x/ count_success}, {error_y/ count_success},{error_z/ count_success}")


bins = [0.0]
for i in range(200):
    bins.append(bins[-1] + 0.005)

rot_bins = [0.0]
for i in range(200):
    rot_bins.append(rot_bins[-1] + 0.0005)

fig, ax = plt.subplots(2)

ax[0].set_xlabel('Error (mm)')
ax[0].set_ylabel('Count')
n, bins, patches = ax[0].hist(np.multiply(error_array_mean[:, :1], 1000), bins=bins)
ax[0].axvline(np.mean(np.multiply(error_array_mean[:, :1], 1000), axis=0), color='k', linestyle='dashed', linewidth=1)
ax[0].axvline(np.mean(np.multiply(error_array_mean_cleaned[:, :1], 1000), axis=0), color='green', linestyle='dashed', linewidth=1)
bins = bins + 0.0025
ax[0].plot(bins[:200], n,linewidth=1, linestyle=(0, (5, 1)), color="black")
ax[0].set_xticks(np.arange(0.0, 1.1, 0.1))
ax[0].set_yticks(np.arange(0.0, 7000., 1000.))

ax[1].set_xlabel('Error (degree)')
ax[1].set_ylabel('Count')
n, bins, patches = ax[1].hist(error_array_mean[:, 1:], bins=rot_bins)
ax[1].axvline(error_array_mean[:, 1:].mean(), color='k', linestyle='dashed', linewidth=1)
ax[1].axvline(error_array_mean_cleaned[:, 1:].mean(), color='green', linestyle='dashed', linewidth=1)
bins = bins + 0.00025
ax[1].plot(bins[:200], n,linewidth=1, linestyle=(0, (5, 1)), color="black")
ax[1].set_xticks(np.arange(0.0, 0.11, 0.01))
ax[1].set_yticks(np.arange(0.0, 5000., 1000.))

fig.tight_layout()
plt.show()