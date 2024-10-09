#  Copyright (c) 2024. Jan-Gerrit Habekost. GNU General Public License. https://www.gnu.org/licenses/gpl-3.0.html.
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
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetPositionIKRequest, GetPositionFKRequest
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


def execute_fk_request(result_queue,  robot_dof, joint_names, robot_eef, process_id, lower_js_limit, upper_js_limit, samples):
    rospy.init_node(f"moveit_ik_experiment_{process_id}", anonymous=True)
    rospy.wait_for_service("/compute_fk")
    moveit_ik = rospy.ServiceProxy("/compute_fk", GetPositionFK)

    lower_js_limit = copy.deepcopy(lower_js_limit)
    upper_js_limit = copy.deepcopy(upper_js_limit)
    samples = copy.deepcopy(samples)

    keep_running = len(result_queue[0]) < samples

    while keep_running:
        #target_pose_input = pose_queue.get(block=True)
        #real_b_np = copy.deepcopy(target_pose_input)
        #del target_pose_input

        in_workspace = False

        random_js = []
        for i in range(robot_dof):
            joint_val = random.uniform(lower_js_limit[i], upper_js_limit[i])
            random_js.append(joint_val)

        # rospy.loginfo("Position: \n%s", correct_poses)
        # rospy.loginfo("Joint State: \n%s", random_js)

        joint_state_msg = JointState()

        joint_state_msg.name = copy.deepcopy(joint_names)
        #print(random_js)
        joint_state_msg.position = random_js
        robot_state_msg = RobotState()
        robot_state_msg.joint_state = joint_state_msg
        robot_state_msg.is_diff = True

        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()

        result = None
        try:

            result = moveit_ik(h, [copy.deepcopy(robot_eef)], robot_state_msg)
            rospy.sleep(0.0001)
        except rospy.ServiceException as exc:
            rospy.logerr("Service did not process request: %s", exc)
            rospy.sleep(0.0001)
            continue

        if result is not None and result.error_code.val == 1:
            pose_msg = result.pose_stamped[0].pose
            pose = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z, pose_msg.orientation.x,
                    pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w, ]
            js = joint_state_msg.position
            result_queue[0].append(pose)
            result_queue[1].append(js)

        keep_running = len(result_queue[0]) < samples
        #print(pose_queue.qsize())
        #print(return_positions)

    del moveit_ik
    rospy.signal_shutdown("Shutting down FK sampler")

def watch_progress(result_queue, nbr_samples):
    progress_bar = tqdm(total=nbr_samples)

    keep_running = len(result_queue[0]) < nbr_samples
    last_val = 0
    while keep_running:
        #print(keep_running)
        curr_val = copy.deepcopy(len(result_queue[0]))
        progress_bar.update(curr_val - last_val)
        last_val = copy.deepcopy(curr_val)
        keep_running = len(result_queue[0]) < nbr_samples
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
    #robot_urdf = config["robot_urdf"]
    robot_eef = config["robot_eef"]
    joint_name_list = config['joint_name']
    upper_js_limit = config['limits']['upper']
    lower_js_limit = config['limits']['lower']

    #move_group = config['move_group']
    #zero_js = config['home_js']

    # Dataset
    #dataset = IKDataset(root=val_data, test=True, robot=args.robot, config=config, mode='val')
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")

    #progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    count_bullshit = 0

    ctx = mp.get_context('forkserver')
    manager = Manager()
    result_queue = manager.list()
    result_queue.append(manager.list())
    result_queue.append(manager.list())

    processes = []

    for i in range(os.cpu_count() - 2):
        process = ctx.Process(target=execute_fk_request, args=(result_queue,  robot_dof, joint_name_list, robot_eef, i, upper_js_limit, lower_js_limit, args.samples))
        process.start()
        processes.append(process)

    watcher_process = ctx.Process(target=watch_progress, args=(result_queue, args.samples))
    watcher_process.start()
    watcher_process.join()
    #watcher_process.terminate()
    #processes.append(watcher_process)

    #print('here')

    for p in processes:
        p.join()
        #p.terminate()
    res = list(result_queue)
    res = [list(result_queue[0][:args.samples]), list(result_queue[1][:args.samples])]
    #print(res)

    with open(f'./data_samplers/results_{args.robot}_{int(args.samples / 1000)}_{args.chain}.p', 'wb') as f:
        pickle.dump(res, f)
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
    parser.add_argument("--cuda", action="store_true", help="Enables cuda")
    parser.add_argument("--manualSeed", type=int,
                        help="Seed for initializing training. (default:none)")
    parser.add_argument("--robot", type=str, default="nicol", help="Robot model IK is trained for")
    parser.add_argument("--gpu", type=int, default=0, help="Robot model IK is trained for")
    parser.add_argument("--chain", type=str, default='', help="chain")
    parser.add_argument("--samples", type=int, default=100000, help="chain")
    args = parser.parse_args()
    print(args)

    main(args)