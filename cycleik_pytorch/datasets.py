import glob
import os, sys
import random
import time
from threading import Thread

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
from .utils import get_kinematic_params
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.transform import Rotation as R
import open3d as o3d
np.set_printoptions(threshold=20000)
np.set_printoptions(suppress=True)


class IKDataset(Dataset):
    def __init__(self, root, mode="train", test=False, robot="nicol", config=None, augmented=False):
        super(IKDataset, self).__init__()
        self.test = test
        self.ground_truth_A = []
        self.ground_truth_B = []
        self.unnormalized_A = []
        self.unnormalized_B = []

        print(robot)
        assert config is not None

        workspace_interval_array, workspace_center_array, limits_upper, limits_lower, normalize_interval_array, normalize_center_array = get_kinematic_params(config)
                
        #print(diff_upper_lower)

        data_path = os.path.dirname(os.path.abspath(__file__))
        print(data_path)

        with open(data_path + "/../data/" +  root + ".p", 'rb') as f:
            loaded_values = pickle.load(f)

        if robot == "fetch" and len(loaded_values[0]) > 1000000:
            with open(data_path + "/../data/" + root + "_1.p", 'rb') as f:
                loaded_values2 = pickle.load(f)

            #print(loaded_values2)

        tmp_nbr_samples = 1000000 if len(loaded_values[0]) > 1000000 else len(loaded_values[0])
        tmp_nbr_samples = 10000 if 15000 > len(loaded_values[0]) > 10000 else tmp_nbr_samples

        #for i in range(tmp_nbr_samples):#len(loaded_values[0])):
        #for i in range(1000):
        #for i in range(tmp_nbr_samples):

        if robot == "fetch" and len(loaded_values[0]) > 1000000:
            loaded_values[0] = loaded_values[0] + loaded_values2[0]
            loaded_values[1] = loaded_values[1] + loaded_values2[1]

        for i in range(len(loaded_values[0])):

            pose = np.array(loaded_values[0][i], dtype=np.float32)
            js = np.array(loaded_values[1][i], dtype=np.float32)

            self.unnormalized_A.append(np.array(js, dtype=np.float32))
            self.unnormalized_B.append(np.array(pose, dtype=np.float32))

            #pose[:3] = np.true_divide(pose[:3], workspace)
            pose[:3] = np.add(pose[:3], workspace_center_array)
            pose[:3] = np.true_divide(pose[:3], workspace_interval_array)

            if i == 0: assert len(limits_upper) == len(limits_lower) == len(js) == config["robot_dof"]

            js = np.add(js, normalize_center_array)
            js = np.true_divide(js, normalize_interval_array)

            self.ground_truth_A.append(np.array(js, dtype=np.float32))
            self.ground_truth_B.append(np.array(pose, dtype=np.float32))

        print("Max. val joint states: {0}".format(np.max(self.unnormalized_A, axis=0)))
        print("Min. val joint states: {0}".format(np.min(self.unnormalized_A, axis=0)))
        print("\n")
        print("Max. val poses: {0}".format(np.max(self.unnormalized_B, axis=0)))
        print("Min. val poses: {0}".format(np.min(self.unnormalized_B, axis=0)))
        print("\n")
        print("Max. val normalized joint states: {0}".format(np.max(self.ground_truth_A, axis=0)))
        print("Min. val normalized joint states: {0}".format(np.min(self.ground_truth_A, axis=0)))
        print("\n")
        print("Max. val normalized poses: {0}".format(np.max(self.ground_truth_B, axis=0)))
        print("Min. val normalized poses: {0}".format(np.min(self.ground_truth_B, axis=0)))

        points = np.array(self.unnormalized_B)[:, :3]

        # Initialize a point cloud object
        pcd = o3d.geometry.PointCloud()
        # Add the points, colors and normals as Vectors
        pcd.points = o3d.utility.Vector3dVector(points)
        workspace_convex_hull, point_indices = pcd.compute_convex_hull()
        workspace_convex_hull.compute_vertex_normals()

        workspace_volume = workspace_convex_hull.get_volume()
        workspace_volume *= 1000000 # cubic meter to cubic centimeter
        print(f'Volume of Action Space: {workspace_volume} cm³')

        workspace_sample_density = len(self.unnormalized_B) / workspace_volume

        print(f'Sample Density: {workspace_sample_density} samples per cm³')

        self.data_size = len(self.ground_truth_B)
        self.normalize_center_array = normalize_center_array
        self.normalize_interval_array = normalize_interval_array
        self.workspace_center_array = workspace_center_array
        self.workspace_interval_array = workspace_interval_array
        print("Size Dataset: ", self.data_size)

    def __getitem__(self, index):
        item_gt_A = self.ground_truth_A[index % len(self.ground_truth_A)]
        item_gt_B = self.ground_truth_B[index % len(self.ground_truth_B)]
        item_real_A = self.unnormalized_A[index % len(self.unnormalized_A)]
        item_real_B = self.unnormalized_B[index % len(self.unnormalized_B)]
        return {"gt_A": item_gt_A, "gt_B": item_gt_B, "real_A": item_real_A, "real_B": item_real_B}

    def __len__(self):
        return len(self.ground_truth_B)

    def get_size(self):
        return self.data_size

    def get_norm_params(self):
        return [self.normalize_center_array, self.normalize_interval_array, self.workspace_center_array, self.workspace_interval_array]


