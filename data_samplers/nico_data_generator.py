#!/usr/bin/env python
from unittest import result
import numpy as np

import random
import time

import pathlib
import xml.etree.ElementTree
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import std_msgs.msg
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
import cv2 as cv

import rospy
import pickle
from tqdm import tqdm
import os

use_platform = False
timeout = 15

rospy.init_node('ik_data_sampler', anonymous=True)

"""  setup  """
rospy.wait_for_service('compute_fk')
calculate_fk = rospy.ServiceProxy('compute_fk', GetPositionFK)

marker_publisher = rospy.Publisher("/visualization_marker", Marker, queue_size=100000)


def main():
    correct_poses = 0
    data = []
    iterations = 200000
    leg = False

    #if os.path.exists('/home/jan-gerrit/repositories/cycleik/data_samplers/results_ur5_1mio.p'):
    #    with open('/home/jan-gerrit/repositories/cycleik/data_samplers/results_ur5_1mio.p',
    #              'rb') as f:
    #        data = pickle.load(f)
    #        f.close()
        # print(data)

    joint_states = []
    poses = []
    jacobians = []

    with tqdm(total=iterations) as pbar:
        while correct_poses < iterations and not rospy.is_shutdown():

            in_workspace = False
            if leg:
                random_js = [random.uniform(-0.6981, 0.6981), random.uniform(-0.4188, 1.2217),
                             random.uniform(-1.5707, 0.5585), random.uniform(-1.5358, 1.2915),
                             random.uniform(-0.7853, 0.7853), random.uniform(-0.7853, 0.7853)]
            else:
                #upper: [0.8, 3.142, 0.314, 0.0, 1.571, 0.872665]
                #lower: [-0.8, -3.142, -1.57, -1.745, -1.571, 0.0]
                random_js = [random.uniform(-0.8, 0.8), random.uniform(-3.142, 3.142),
                             random.uniform(-1.57, 0.314), random.uniform(-1.745, 0.0),
                             random.uniform(-1.571, 1.571), random.uniform(0.0, 0.872665)]

            # rospy.loginfo("Position: \n%s", correct_poses)
            # rospy.loginfo("Joint State: \n%s", random_js)

            joint_state_msg = JointState()
            if leg:
                joint_state_msg.name = ["r_hip_z", "r_hip_x", "r_hip_y", "r_knee_y", "r_ankle_y", "r_ankle_x"]
            else:
                joint_state_msg.name = ["l_shoulder_z", "l_shoulder_y", "l_arm_x", "l_elbow_y", "l_wrist_z", "l_wrist_x"]
                print(random_js)
            joint_state_msg.position = random_js
            robot_state_msg = RobotState()
            robot_state_msg.joint_state = joint_state_msg
            robot_state_msg.is_diff = True

            h = std_msgs.msg.Header()
            h.stamp = rospy.Time.now()

            result = None
            try:
                if leg:
                    result = calculate_fk(h, ["right_foot:11"], robot_state_msg)
                else:
                    result = calculate_fk(h, ["left_tcp"], robot_state_msg)
            except rospy.ServiceException as exc:
                rospy.logerr("Service did not process request: %s", exc)
                rospy.sleep(0.00000001)
                continue

            #print(result.error_code.val)

            if result is not None and (result.error_code.val == -10 or result.error_code.val == -12):
                # rospy.logerr("Joint State in collision")
                rospy.sleep(0.000000001)
                continue

            if result is not None:
                if result.error_code.val == 1:
                    correct_poses += 1
                    # rospy.loginfo("Call successfull. Pose: \n%s", result.pose_stamped[0].pose)

                    pose_msg = result.pose_stamped[0].pose
                    pose = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z, pose_msg.orientation.x,
                            pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w, ]
                    js = joint_state_msg.position
                    #print(pose)
                    poses.append(pose)
                    joint_states.append(js)

                    data.append([result.pose_stamped[0].pose, joint_state_msg])
                    pbar.update(1)

                    # print(pose)
                    marker = Marker(
                        type=Marker.SPHERE,
                        action=Marker.ADD,
                        id=correct_poses,
                        lifetime=rospy.Duration(),
                        pose=Pose(Point(pose[0], pose[1], pose[2]), Quaternion(0, 0, 0, 1)),
                        scale=Vector3(0.001, 0.001, 0.001),
                        header=Header(frame_id='torso:11'),
                        color=ColorRGBA(0.0, 1.0, 0.0, 0.8),
                        # lifetime=0,
                        frame_locked=False)
                    marker_publisher.publish(marker)
                    rospy.sleep(0.001)
            # else:
            # rospy.loginfo("Call unsuccessfull")

            rospy.sleep(0.00000001)

    data = [poses, joint_states]

    with open('./data/results_nico_200_3_left.p', 'wb') as f:
        pickle.dump(data, f)
        f.close()


if __name__ == '__main__':
    main()