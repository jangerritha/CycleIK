#  Copyright (c) 2024. Jan-Gerrit Habekost. GNU General Public License. https://www.gnu.org/licenses/gpl-3.0.html.
import numpy as np
import torch

from cycleik_pytorch import CycleIK
import argparse
from moveit_msgs.msg import DisplayRobotState, RobotState, DisplayTrajectory, RobotTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cycleik_pytorch import load_config

import rospy


def display_redundancy(pose_np, z, js_samples, display_trajectory_msg, cycleik, config, t):

    joint_distribution = cycleik.inverse_kinematics_distribution(pose=pose_np, noise=z, js_samples=js_samples,
                                                                 calculate_error=True)

    #for i in range(0):
    #    joint_distribution = joint_distribution.detach().cpu().numpy()
    #    min_noise = np.min(joint_distribution, axis=0)
    #    max_noise = np.max(joint_distribution, axis=0)
    #    joint_distribution = np.random.uniform(low=min_noise, high=max_noise,
    #                                           size=(js_samples, config['robot_dof'])) * 1.5
    #    joint_distribution = np.clip(a=joint_distribution, a_min=-1, a_max=1)
    #
    #    joint_distribution = cycleik.inverse_kinematics_distribution(pose=pose_np, noise=joint_distribution,
    #                                                                 js_samples=js_samples,
    #                                                                 calculate_error=True)
    #
    print(joint_distribution)
    display_trajectory_msg.trajectory.append(RobotTrajectory())
    display_trajectory_msg.trajectory[t].joint_trajectory = JointTrajectory()
    display_trajectory_msg.trajectory[t].joint_trajectory.joint_names = config["joint_name"]

    for e, joint_position in enumerate(joint_distribution):
        trajectory_point = JointTrajectoryPoint()
        trajectory_point.positions = list(joint_position)

        display_trajectory_msg.trajectory[t].joint_trajectory.points.append(trajectory_point)

    return display_trajectory_msg

def display_simple_path(js, display_trajectory_msg, config, t):

    display_trajectory_msg.trajectory.append(RobotTrajectory())
    display_trajectory_msg.trajectory[t].joint_trajectory = JointTrajectory()
    display_trajectory_msg.trajectory[t].joint_trajectory.joint_names = config["joint_name"]

    trajectory_point = JointTrajectoryPoint()
    trajectory_point.positions = list(js)

    display_trajectory_msg.trajectory[t].joint_trajectory.points.append(trajectory_point)

    return display_trajectory_msg


def main(args):

    ros_node = rospy.init_node("cycleik_vis", anonymous=True)

    print(args.robot)

    config = load_config(args.robot)
    config = config[f"{args.chain}"]
    if args.network == 'GAN':
        js_samples = config['debug']['null_space_samples']

    nbr_points = config['debug']['points']

    cycleik = CycleIK(robot=args.robot, cuda_device=args.gpu, chain=args.chain, verbose=True)

    display_pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=10)

    start_pose_np = np.array(config['debug']['pose'])
    position_axis = 0
    if config['debug']['axis'] == 'X':
        position_axis = 0
    elif config['debug']['axis'] == 'Y':
        position_axis = 1
    elif config['debug']['axis'] == 'Z':
        position_axis = 2
    else:
        raise NotImplementedError('You specified a wrong value for the debug pose under config/{args.robot}.yaml,'
                                  ' coose from the following: \'X\', \'Y\' or \'Z\'')
    #start_pose_np = np.array([0.3, 0.0, 0.5, -1., 0., 0., 0.])
    start_pose_np = np.reshape(start_pose_np, newshape=(1, 7))

    joint_state_msg = JointState()
    joint_state_msg.name = config["joint_name"]
    joint_state_msg.position = list(np.zeros(config['robot_dof']))

    robot_state_msg = RobotState()
    robot_state_msg.joint_state = joint_state_msg
    robot_state_msg.is_diff = True

    display_trajectory_msg = DisplayTrajectory()
    display_trajectory_msg.trajectory_start = robot_state_msg

    print("\n\n heeree")

    if args.network == 'GAN':
        for t in range(nbr_points):

            pose_np = np.copy(start_pose_np)
            #pose_np[0, 0] -= t * 0.05
            pose_np[0, position_axis] -= t * 0.05
            pose_np = np.repeat(pose_np, js_samples, axis=0)

            z = np.random.uniform(low=-1.0, high=1.0, size=(js_samples, config['robot_dof']))

            display_trajectory_msg = display_redundancy(pose_np, z, js_samples, display_trajectory_msg, cycleik, config, t)
    else:
        print("\n\n heeree")
        pose_np = np.copy(start_pose_np)
        joint_distribution = None
        for t in range(1, nbr_points):
            next_pose = np.copy(start_pose_np)
            next_pose[:,position_axis] = np.copy(start_pose_np)[:,position_axis] + (t * 0.05)
            pose_np = np.concatenate((pose_np, next_pose), axis=0)

        joint_distribution, _, _, _ = cycleik.inverse_kinematics(poses=pose_np, calculate_error=args.calculate_error)

        print(joint_distribution)

        for t, js in enumerate(joint_distribution):
            display_trajectory_msg = display_simple_path(js, display_trajectory_msg, config, t)






            #joint_state_msg = JointState()
            #joint_state_msg.name =
            #joint_state_msg.position = list(joint_position)
            #
            #robot_state_msg = RobotState()
            #robot_state_msg.joint_state = joint_state_msg
            #robot_state_msg.is_diff = True
            #
            #display_state_msg = DisplayRobotState()
            #display_state_msg.state = robot_state_msg
    display_pub.publish(display_trajectory_msg)
            #
    rospy.sleep(1)
    display_pub.publish(display_trajectory_msg)
    #
    rospy.sleep(1)
    #display_pub.publish(display_trajectory_msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
    parser.add_argument("--cuda", action="store_true", help="Enables cuda")
    parser.add_argument("--calculate_error", action="store_true", help="Enables cuda")
    parser.add_argument("--use_ga", action="store_true", help="Enables cuda")
    parser.add_argument("--use_optimizer", action="store_true", help="Enables cuda")
    parser.add_argument("--manualSeed", type=int,
                        help="Seed for initializing training. (default:none)")
    parser.add_argument("--robot", type=str, default="nicol", help="Robot model IK is trained for")
    parser.add_argument("--gpu", type=int, default=0, help="Robot model IK is trained for")
    parser.add_argument("--network", type=str, default="MLP", help="Robot model IK is trained for")
    parser.add_argument("--core-model", type=str, default="nicol", help="Robot model IK is trained for")
    parser.add_argument("--autoencoder", action="store_true", help="Enables learned FK")
    parser.add_argument("--two-stage", action="store_true", help="Enables two-stage learned FK training")
    parser.add_argument("--finetune", action="store_true", help="Enables two-stage learned FK training")
    parser.add_argument("--chain", type=str, default="right_arm", help="Robot model Kinematic Chain")

    #print(args)

    test_args = parser.parse_args()
    main(test_args)