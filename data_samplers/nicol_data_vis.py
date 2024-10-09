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
from visualization_msgs.msg import Marker, MarkerArray
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
rospy.wait_for_service('moveit/compute_fk')
calculate_fk = rospy.ServiceProxy('moveit/compute_fk', GetPositionFK)

marker_publisher = rospy.Publisher("moveit/visualization_marker_array", MarkerArray, queue_size=2)

def main():
    root = "results_nicol_1000_4"

    data_path = str(os.getcwd())
    print(data_path)

    with open(data_path + "/data/" + root + ".p", 'rb') as f:
        loaded_values = pickle.load(f)

    marker_array = MarkerArray()
    markers = []

    for i in tqdm(range(len(loaded_values[0]))):
        pose = np.array(loaded_values[0][i], dtype=np.float32)
        js = np.array(loaded_values[1][i], dtype=np.float32)

        # print(pose)
        marker = Marker(
            type=Marker.SPHERE,
            action=Marker.ADD,
            id=i,
            lifetime=rospy.Duration(),
            pose=Pose(Point(pose[0], pose[1], pose[2]), Quaternion(0, 0, 0, 1)),
            scale=Vector3(0.001, 0.001, 0.001),
            header=Header(frame_id='world'),
            color=ColorRGBA(0.0, 1.0, 0.0, 0.8),
            # lifetime=0,
            frame_locked=False)

        marker_array.markers.append(marker)

        #marker_publisher.publish(marker)
        #rospy.sleep(0.00001)

    marker_publisher.publish(marker_array)


if __name__ == '__main__':
    main()