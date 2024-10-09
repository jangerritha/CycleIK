#  Copyright (c) 2024. Jan-Gerrit Habekost. GNU General Public License. https://www.gnu.org/licenses/gpl-3.0.html.
import argparse
import copy
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
import os
from cycleik_pytorch import load_config, IKDataset
import pytorch_kinematics as pk
import time
import matplotlib.pyplot as plt
import pandas as ps
import multiprocessing as mp
from multiprocessing import Manager
import rospy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetPositionIKRequest
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
import std_msgs.msg
import moveit_msgs.msg
import moveit_msgs.srv
import bio_ik_msgs.msg
import bio_ik_msgs.srv
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
import pickle
np.set_printoptions(precision=32)

def execute_ik_request(pose_queue, result_queue,  robot_dof, joint_names, robot_eef, group_name, zero_js, process_id):
    rospy.init_node(f"moveit_ik_experiment_{process_id}", anonymous=True)
    rospy.wait_for_service("/compute_ik")
    moveit_ik = rospy.ServiceProxy("/compute_ik", GetPositionIK)

    keep_running = pose_queue.qsize() > 0

    while keep_running:
        target_pose_input = pose_queue.get(block=True)
        real_b_np = copy.deepcopy(target_pose_input)
        del target_pose_input
        request = GetPositionIKRequest()
        #print(request)
        request.ik_request.group_name = copy.deepcopy(group_name)
        #request.ik_request.timeout.nsecs = 5000000
        request.ik_request.timeout.nsecs = 1000000
        request.ik_request.avoid_collisions = False
        request.ik_request.ik_link_name = copy.deepcopy(robot_eef)

        current_state = RobotState()
        current_state.is_diff = True
        #print(joint_names)
        #print(zero_js)
        current_state.joint_state.name = copy.deepcopy(joint_names)
        current_state.joint_state.position = copy.deepcopy(zero_js)
        request.ik_request.robot_state = current_state

        target_pose = PoseStamped()
        target_pose.pose.position.x = real_b_np[0]
        target_pose.pose.position.y = real_b_np[1]
        target_pose.pose.position.z = real_b_np[2]
        target_pose.pose.orientation.x = real_b_np[3]
        target_pose.pose.orientation.y = real_b_np[4]
        target_pose.pose.orientation.z = real_b_np[5]
        target_pose.pose.orientation.w = real_b_np[6]
        request.ik_request.pose_stamped = target_pose

        response = moveit_ik(request)
        rospy.sleep(0.001)
        return_positions = []
        if response.error_code.val != 1:
            for i in range(robot_dof):
                return_positions.append(-10)
        else:
            all_names = response.solution.joint_state.name
            full_js = response.solution.joint_state.position
            selected_js_unordered = []
            selected_names_unordered = []
            for i in range(len(full_js)):
                curr_name = all_names[i]
                if curr_name in joint_names:
                    selected_names_unordered.append(curr_name)
                    selected_js_unordered.append(full_js[i])
            for joint_name in joint_names:
                for e, unordered_name in enumerate(selected_names_unordered):
                    if unordered_name == joint_name:
                        return_positions.append(selected_js_unordered[e])
        keep_running = pose_queue.qsize() > 0
        #print(pose_queue.qsize())
        #print(return_positions)
        assert len(return_positions) == robot_dof
        result_queue.append([real_b_np, return_positions])
        #rospy.spin_once()
        del request
        del response
    del moveit_ik
    rospy.signal_shutdown("Shutting down IK test")

def watch_progress(result_queue, nbr_samples):
    progress_bar = tqdm(total=nbr_samples)

    keep_running = len(result_queue) < nbr_samples
    last_val = 0
    while keep_running:
        #print(keep_running)
        curr_val = copy.deepcopy(len(result_queue))
        progress_bar.update(curr_val - last_val)
        last_val = copy.deepcopy(curr_val)
        keep_running = len(result_queue) < nbr_samples
        time.sleep(2)

    del progress_bar

def main(args):

    torch.cuda.empty_cache()

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    config = load_config(args.robot)[f'{args.chain}']

    val_data = config["val_data"]
    robot_dof = config["robot_dof"]
    robot_urdf = config["robot_urdf"]
    robot_eef = config["robot_eef"]
    joint_name_list = config['joint_name']
    move_group = config['move_group']
    zero_js = config['home_js']

    try:
        os.makedirs(f"results/IROS/precision/{args.robot}/")
    except OSError:
        pass

    # Dataset
    dataset = IKDataset(root=val_data, test=True, robot=args.robot, config=config, mode='val')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    count_bullshit = 0

    ctx = mp.get_context('forkserver')
    pose_queue = ctx.Queue()
    manager = Manager()
    result_queue = manager.list()

    for i, data in progress_bar:#for i, data in progress_bar
        # get batch size data
        real_B = data["real_B"]
        #print(real_images_B.cpu().detach().numpy().shape)
        #print("-----------------------------------------------")

        #print(real_B)
        real_b_np = real_B.clone().numpy()[0]
        #print(real_b_np.shape)
        real_b_np = list(real_b_np)
        pose_queue.put(copy.deepcopy(real_b_np))

    processes = []

    for i in range(os.cpu_count() - 2):
        process = ctx.Process(target=execute_ik_request, args=(pose_queue, result_queue,  robot_dof, joint_name_list, robot_eef, move_group, zero_js, i))
        process.start()
        processes.append(process)

    watcher_process = ctx.Process(target=watch_progress, args=(result_queue, pose_queue.qsize()))
    watcher_process.start()
    watcher_process.join()
    #watcher_process.terminate()
    #processes.append(watcher_process)

    #print('here')

    for p in processes:
        p.join()
        #p.terminate()
    res = list(result_queue)
    #print(res)

    with open(rf"results/IROS/precision/{args.robot}/{args.ik_name}.p", "wb") as output_file:
        pickle.dump(res, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
    parser.add_argument("--cuda", action="store_true", help="Enables cuda")
    parser.add_argument("--manualSeed", type=int,
                        help="Seed for initializing training. (default:none)")
    parser.add_argument("--robot", type=str, default="nicol", help="Robot model IK is trained for")
    parser.add_argument("--gpu", type=int, default=0, help="Robot model IK is trained for")
    parser.add_argument("--chain", type=str, default='', help="chain")
    parser.add_argument("--ik_name", type=str, default='', help="chain")
    args = parser.parse_args()
    print(args)

    main(args)