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

            random_js = [random.uniform(-2.8973, 2.8973), random.uniform(-1.7628, 1.7628),
                         random.uniform(-2.8973, 2.8973), random.uniform(-3.0718, -0.0698),
                         random.uniform(-2.8973, 2.8973), random.uniform(-0.0175, 3.7525),
                         random.uniform(-2.8973, 2.8973)]

            # rospy.loginfo("Position: \n%s", correct_poses)
            # rospy.loginfo("Joint State: \n%s", random_js)

            joint_state_msg = JointState()
            joint_state_msg.name = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
            joint_state_msg.position = random_js
            robot_state_msg = RobotState()
            robot_state_msg.joint_state = joint_state_msg
            robot_state_msg.is_diff = True

            h = std_msgs.msg.Header()
            h.stamp = rospy.Time.now()

            result = None
            try:
                result = calculate_fk(h, ["panda_hand_tcp"], robot_state_msg)
            except rospy.ServiceException as exc:
                rospy.logerr("Service did not process request: %s", exc)
                rospy.sleep(0.00001)
                continue

            if result is not None and (result.error_code.val == -10 or result.error_code.val == -12):
                # rospy.logerr("Joint State in collision")
                rospy.sleep(0.00001)
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
                        header=Header(frame_id='world'),
                        color=ColorRGBA(0.0, 1.0, 0.0, 0.8),
                        # lifetime=0,
                        frame_locked=False)
                    marker_publisher.publish(marker)
                    rospy.sleep(0.001)
            # else:
            # rospy.loginfo("Call unsuccessfull")

            rospy.sleep(0.00001)

    data = [poses, joint_states]

    with open('./data/results_panda_200_4.p', 'wb') as f:
        pickle.dump(data, f)
        f.close()


if __name__ == '__main__':
    main()