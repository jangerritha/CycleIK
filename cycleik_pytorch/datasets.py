# Copyright (c) 2023. Jan-Gerrit Habekost. GNU General Public License https://www.gnu.org/licenses/gpl-3.0.html.

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

        data_path = str(os.getcwd())
        print(data_path)

        with open(data_path + "/../cycleik/data/" +  root + ".p", 'rb') as f:
            loaded_values = pickle.load(f)

        tmp_nbr_samples = 1000000 if len(loaded_values[0]) > 1000000 else len(loaded_values[0])
        tmp_nbr_samples = 10000 if 15000 > len(loaded_values[0]) > 10000 else tmp_nbr_samples

        #for i in range(tmp_nbr_samples):#len(loaded_values[0])):
        #for i in range(1000):
        #for i in range(tmp_nbr_samples):
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
        #print(np.max(self.unnormalized_A, axis=0))
        #print(np.max(self.unnormalized_B, axis=0))
        #print("\n")
        #print(np.max(np.abs(self.ground_truth_A), axis=0))
        #print(np.max(np.abs(self.ground_truth_B), axis=0))

        #if augmented:

        #self.complex_hull = ConvexHull(points=np.array(self.unnormalized_B)[:, :3])
        augmented_points = []

        # Voxel builder callback function
        def build_voxels(voxel_grid, voxels_all):
            vox_mesh = o3d.geometry.TriangleMesh()
            # check if the counter is more than the amount of voxels
            for params_counter in range(len(voxels_all)):
                # get the size of the voxels
                voxel_size = voxel_grid.voxel_size
                # create a box primitive of size 1x1x1
                cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
                # paint the box uniformly with the color of the voxel
                cube.paint_uniform_color(voxels_all[params_counter].color)
                # scale the box to the size of the voxel
                cube.scale(voxel_size, center=cube.get_center())
                # get the center position of the current voxel
                voxel_center = voxel_grid.get_voxel_center_coordinate(voxels_all[params_counter].grid_index)
                # translate the box to the voxel center
                cube.translate(voxel_center, relative=False)
                # add the box primitive to the voxel mesh
                vox_mesh += cube

                return vox_mesh
                # on the first loop create the geometry and on subsequent iterations update the geometry
                #if params.counter == 0:
                    #vis.add_geometry(params.vox_mesh)
                #else:
                    #vis.update_geometry(params.vox_mesh)

                # update the renderer
                #vis.update_renderer()
                # tick up the counter
                #params.counter += 1

        # Read the bunny statue point cloud using numpy's loadtxt
        # Separate the into points, colors and normals array
        points = np.array(self.unnormalized_B)[:, :3]
        #colors = point_cloud[:, 3:6]
        #normals = point_cloud[:, 6:]

        # Initialize a point cloud object
        pcd = o3d.geometry.PointCloud()
        # Add the points, colors and normals as Vectors
        pcd.points = o3d.utility.Vector3dVector(points)
        #pcd.colors = o3d.utility.Vector3dVector(colors)
        #pcd.normals = o3d.utility.Vector3dVector(normals)
        # Create a voxel grid from the point cloud with a voxel_size of 0.01
        #voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.0025)
        # Get all the voxels in the voxel grid
        #voxels_all = voxel_grid.get_voxels()

        # get all the centers and colors from the voxels in the voxel grid
        #all_centers = []
        #all_colors = []
        #for e, voxel in tqdm(enumerate(voxels_all)):
        #    voxel_center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
            #aug_pose = np.copy(self.ground_truth_B[e])
            #aug_pose[:3] = voxel_center
            #augmented_points.append(aug_pose)
        #    all_centers.append(voxel_center)
        #    all_colors.append(voxel.color)

        #triangle_mesh = build_voxels(voxel_grid, voxels_all)

        #alpha = 0.03
        #print(f"alpha={alpha:.3f}")
        workspace_convex_hull, point_indices = pcd.compute_convex_hull()
        #triangle_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        workspace_convex_hull.compute_vertex_normals()
        #print(workspace_convex_hull.vertices)

        workspace_volume = workspace_convex_hull.get_volume()
        workspace_volume *= 1000000 # cubic meter to cubic centimeter
        print(f'Volume of Action Space: {workspace_volume} cm³')

        workspace_sample_density = len(self.unnormalized_B) / workspace_volume

        print(f'Sample Density: {workspace_sample_density} samples per cm³')

        # Initialize a visualizer object
        #vis = o3d.visualization.Visualizer()
        # Create a window, name it and scale it
        #vis.create_window(window_name='NICOL Visualize', width=800, height=600)
        # Create a point cloud that will contain the voxel centers as points
        #pcd_centers = o3d.geometry.PointCloud()
        # Tranform the numpy array into points for the point cloud
        #pcd_centers.points = o3d.utility.Vector3dVector(all_centers)
        # pcd_centers.colors = o3d.utility.Vector3dVector(all_colors)'
        #pcd_centers.paint_uniform_color([1, 0, 0])
        # Add the voxel centers point cloud to the visualizer
        #vis.add_geometry(pcd_centers)

        # Invoke the callback function
        #vis.register_animation_callback(build_voxels)
        # We run the visualizater
        #vis.run()
        # Once the visualizer is closed destroy the window and clean up
        #vis.destroy_window()

            #self.ground_truth_B = augmented_points
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


