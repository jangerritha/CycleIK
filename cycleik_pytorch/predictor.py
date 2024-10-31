import random

from .models import GenericGenerator, GenericNoisyGenerator
from .utils import load_config, get_kinematic_params, slice_fk_pose, normalize_pose, renormalize_pose, renormalize_joint_state
import pytorch_kinematics as pk
import numpy as np
import torch
import time
import os
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from scipy.spatial.transform import Rotation as R
import gdown
from .Bezier import Bezier
from scipy.spatial.transform import Slerp

class CycleIK:

    def __init__(self, robot, cuda_device=None, verbose=False, chain="", nicol_compatibility_mode=False, use_gan=False):
        assert robot is not None and type(robot) == str, "Robot name must be set as string with valid config in path"

        data_path = os.path.dirname(os.path.abspath(__file__))
        print(f"data path: {data_path}")

        try:
            os.makedirs(data_path + "/../weights")
        except OSError:
            pass

        try:
            os.makedirs(data_path + f"/../weights/{robot}")
        except OSError:
            pass

        self.nicol_compatibility_mode = nicol_compatibility_mode

        self.robot = robot
        self.device_name = "cpu" if cuda_device is None else f"cuda:{cuda_device}"
        self.device = torch.device(self.device_name)
        self.config = load_config(robot)[f'{chain}']
        self.verbose = verbose
        self.chain = chain
        self.use_gan = use_gan

        self.robot_dof   = self.config["robot_dof"]
        self.robot_urdf  = self.config["robot_urdf"]
        self.robot_eef   = self.config["robot_eef"]

        #self.physicsClient = p.connect(p.DIRECT)
        ##self.pybullet_robot = p.loadURDF(f"../assets/urdf/{self.robot_urdf}", flags=p.URDF_USE_SELF_COLLISION)
        #self.pybullet_robot = p.loadURDF(f"/home/jan-gerrit/repositories/cycleik/assets/urdf/NICOL.urdf", flags=p.URDF_USE_SELF_COLLISION)
        ## nbr_joints = p.getNumJoints(robot)
        #self.pybullet_joint_ids = [35, 36, 37, 38, 39, 40, 41, 42]
        #self.collision_check_initial_position = [1.5, 0., 0., 0., 0., 0., 0., 0.]
        #p.setTimeStep(100, self.pybullet_robot)

        workspace_interval_array, workspace_center_array, limits_upper, limits_lower, normalize_interval_array, normalize_center_array = get_kinematic_params(self.config)
        self.workspace_interval_array   = torch.Tensor(workspace_interval_array).to(self.device)
        self.workspace_center_array     = torch.Tensor(workspace_center_array).to(self.device)
        self.limits_upper               = torch.Tensor(limits_upper).to(self.device)
        self.limits_lower               = torch.Tensor(limits_lower).to(self.device)
        self.normalize_interval_array   = torch.Tensor(normalize_interval_array).to(self.device)
        self.normalize_center_array     = torch.Tensor(normalize_center_array).to(self.device)

        self.model = GenericGenerator(input_size=7, output_size=self.robot_dof,
                                    nbr_tanh=self.config["IKNet"]["architecture"]["nbr_tanh"],
                                    activation=self.config["IKNet"]["architecture"]["activation"],
                                    layers=self.config["IKNet"]["architecture"]["layers"]).to(self.device)
        if self.config["robot_dof"] > 6 and self.use_gan:
            self.gan = GenericNoisyGenerator(input_size=7, output_size=self.robot_dof,
                                          noise_vector_size=self.config["GAN"]["architecture"]["noise_vector_size"],
                                          nbr_tanh=self.config["GAN"]["architecture"]["nbr_tanh"],
                                          activation=self.config["GAN"]["architecture"]["activation"],
                                          layers=self.config["GAN"]["architecture"]["layers"]).to(self.device)

        data_path = str(Path(__file__).parent.parent.absolute())
        try:
            model_path = os.path.join(data_path, "weights", str(self.robot), f"model_MLP_with_kinematics_{self.chain}.pth")
            print(f"Model path: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError as e:
            print("\n\n FileNotFoundError, did not load CycleIK Network weights! You probably have the wrong checkpoint name. \n\n")

            if 'pretrained_weights' in self.config['IKNet'].keys():
                url = self.config['IKNet']['pretrained_weights']
                output = os.path.dirname(os.path.abspath(__file__))
                output = os.path.join(output, f'../weights/{self.robot}/model_MLP_with_kinematics_{self.chain}.pth')
                print("Found pretrained weights, attempting to download them")
                gdown.download(url, output, quiet=False, fuzzy=True)

            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except FileNotFoundError as e:
                print("\n\n FileNotFoundError, did not load CycleIK Network weights! You probably have the wrong checkpoint name. \n\n")

        except RuntimeError as e:
            print("\n\n Runtime Warning, did not load CycleIK Network weights! \n\n")
            raise e

        self.model.eval()

        if self.config["robot_dof"] > 6  and self.use_gan:
            try:
                self.gan.load_state_dict(torch.load(os.path.join(data_path, "weights", str(self.robot), f"model_GAN_with_kinematics_{self.chain}.pth"), map_location=self.device))
            except FileNotFoundError as e:
                #print("\n\n FileNotFoundError, did not load CycleIK Network weights! You probably have the wrong checkpoint name. \n\n")
                pass
            except RuntimeError as e:
                #print("\n\n Runtime Warning, did not load CycleIK Network weights! \n\n")
                pass
            self.gan.eval()



        urdf_path = os.path.join(data_path, self.robot_urdf)
        chain = pk.build_serial_chain_from_urdf(open(urdf_path).read(), self.robot_eef)
        self.chain = chain.to(dtype=torch.float32, device=self.device)

    def __del__(self):
        #p.disconnect()
        pass

    def get_random_samples(self, batch_size):
        random_js = (torch.rand((batch_size, self.robot_dof)).to(self.device) - 0.5) * 2
        random_js = torch.mul(random_js, self.normalize_interval_array.repeat(batch_size, 1))
        random_js = torch.add(random_js, self.normalize_center_array.repeat(batch_size, 1))
        pose_for_js = self.chain.forward_kinematics(random_js)
        pos = torch.reshape(pose_for_js.get_matrix()[:, :3, 3:], shape=(batch_size, 3))
        rot = pk.matrix_to_quaternion(pose_for_js.get_matrix()[:, :3, :3])
        rot = torch.concat((rot[:, 1:], rot[:, :1]), dim=1)
        pose_for_js = torch.concat((pos, rot), dim=1)
        pose_for_js = pose_for_js.detach().cpu().numpy()
        random_js = random_js.detach().cpu().numpy()

        #res = self.check_collision(random_js)

        poses, js = [], []

        for i in tqdm(range(len(pose_for_js)), total=len(pose_for_js)):
            pose = []
            for j in range(len(pose_for_js[i])):
                pose.append(pose_for_js[i][j])
            poses.append(pose)
            js_temp = []
            for j in range(len(random_js[i])):
                js_temp.append(random_js[i][j])
            js.append(js_temp)
        return poses, js

    @torch.no_grad()
    def inverse_kinematics(self, poses: np.ndarray, calculate_error = False):
        st = time.time()
        poses = torch.Tensor(poses).to(self.device)
        unnormalized_poses = poses.clone()
        bs = len(poses)

        poses = normalize_pose(poses, bs, workspace_move=self.workspace_center_array, workspace_renormalize=self.workspace_interval_array, slice_pk_result=False)
        js_result = self.model(poses)
        js_result = renormalize_joint_state(js_result, bs, single_renormalize_move=self.normalize_center_array, single_renormalize=self.normalize_interval_array)
        #js_result = torch.mul(js_result, self.normalize_interval_array.repeat(bs, 1))
        #js_result = torch.add(js_result, self.normalize_center_array.repeat(bs, 1))

        et = time.time()
        elapsed_time_ik = et - st

        error = None
        elapsed_time_fk = None
        if calculate_error:
            st = time.time()
            pose_for_js = self.chain.forward_kinematics(js_result)
            forward_result = slice_fk_pose(pose_for_js, batch_size=bs)

            et = time.time()
            elapsed_time_fk = et - st
            print(f'PK Pose: {forward_result}')

            unnormalized_poses_rot = unnormalized_poses.detach().cpu().numpy()
            reached_poses_rot = forward_result.detach().cpu().numpy()
            r_target = R.from_quat(list(unnormalized_poses_rot[:, 3:]))
            r_result = R.from_quat(list(reached_poses_rot[:, 3:]))
            # print(r_result)
            rotation_diff = r_target * r_result.inv()
            rotation_diff_save = rotation_diff.as_euler('zyx', degrees=True)
            rotation_diff = np.abs(rotation_diff_save)

            error = torch.abs(torch.concat((torch.subtract(forward_result[:, :3], unnormalized_poses[:, :3]), torch.Tensor(rotation_diff).to(self.device)), dim=1)).detach().cpu().numpy()

        if self.nicol_compatibility_mode:
            js_result[:, 6:8] *= 4

        js_result = js_result.detach().cpu().numpy()

        if self.verbose:
            print(f'Execution time IK: {elapsed_time_ik} seconds')
            if calculate_error:
                print(f'Execution time FK: {elapsed_time_fk} seconds')
                print(f'Average FK Error: {error}')

        return js_result, error, elapsed_time_ik, elapsed_time_fk

    @torch.no_grad()
    def inverse_kinematics_distribution_old(self, pose: np.ndarray, js_samples = 1000, calculate_error=False):
        st = time.time()
        noise_vector_size = self.config["GAN"]["architecture"]["noise_vector_size"]
        pose = torch.Tensor(pose).reshape((1, 7)).to(self.device)
        unnormalized_pose = pose.clone()

        pose = normalize_pose(pose, batch_size=1, workspace_move=self.workspace_center_array,
                               workspace_renormalize=self.workspace_interval_array, slice_pk_result=False)
        noise_z = torch.rand((js_samples, noise_vector_size)).to(self.device)

        js_result = self.gan(noise_z, pose.repeat((js_samples, 1)))
        js_result = renormalize_joint_state(js_result, batch_size=js_samples, single_renormalize_move=self.normalize_center_array, single_renormalize=self.normalize_interval_array)
        # js_result = torch.mul(js_result, self.normalize_interval_array.repeat(bs, 1))
        # js_result = torch.add(js_result, self.normalize_center_array.repeat(bs, 1))

        et = time.time()
        elapsed_time_ik = et - st

        def sort_the_array_column_wise(arr):
            for j in range (len(arr[0])):
                for i in range(len(arr) - 1):
                    if arr[i][j] < arr[i + 1][j]:
                        temp = arr[i][j]
                        arr[i][j] = arr[i + 1][j]
                        arr[i + 1][j] = temp

            for i in range(len(arr)):

                for j in range(len(arr[0])):

                    print(arr[i][j], end=" ")

                print()

        size = len(js_result)
        print("The array before performing sorting operation is: ")
        for i in range(len(js_result)):
            for j in range(len(js_result[0])):

                print(js_result[i][j], end=" ")
            print()

        print("The array after performing sorting operation is: ")
        sort_the_array_column_wise(js_result)

        error = None
        elapsed_time_fk = None
        if calculate_error:
            st = time.time()
            pose_for_js = self.chain.forward_kinematics(js_result)
            forward_result = slice_fk_pose(pose_for_js, batch_size=js_samples)

            et = time.time()
            elapsed_time_fk = et - st
            print(f'PK Pose: {forward_result}')
            error = torch.abs(torch.subtract(forward_result, unnormalized_pose.repeat((js_samples, 1)))).detach().cpu().numpy()

        js_result = js_result.detach().cpu().numpy()

        if self.verbose:
            print(f'Execution time IK: {elapsed_time_ik} seconds')
            if calculate_error:
                print(f'Execution time FK: {elapsed_time_fk} seconds')
                print(f'Average FK Error: {error}')

        return js_result, error, elapsed_time_ik, elapsed_time_fk

    @torch.no_grad()
    def inverse_kinematics_distribution(self, pose: np.ndarray, noise: np.ndarray, js_samples=1000, calculate_error=False):
        assert len(pose.shape) == 2, "Input pose to inverse_kinematics_distribution() function should have shape (batch_size, 7)"

        noise_vector_size = self.config["GAN"]["architecture"]["noise_vector_size"]

        assert len(noise.shape) == 2, "Input noise to inverse_kinematics_distribution() function should have shape (batch_size, noise_vector_size). Look up the noise_vector_size in the config"

        assert noise.shape[1] == noise_vector_size, "noise vectors in the noise batch should have shape (batch_size, noise_vector_size). Look up the noise_vector_size in the config"

        assert len(noise) == len(pose), "mismatch between batch_size of pose and noise input"

        pose = torch.Tensor(pose).to(self.device)
        noise = torch.Tensor(noise).to(self.device)

        pose = normalize_pose(pose, batch_size=len(pose), workspace_move=self.workspace_center_array,
                              workspace_renormalize=self.workspace_interval_array, slice_pk_result=False)

        js_result = self.gan(noise, pose)
        js_result = renormalize_joint_state(js_result, batch_size=len(js_result),
                                            single_renormalize_move=self.normalize_center_array,
                                            single_renormalize=self.normalize_interval_array)
        # js_result = torch.mul(js_result, self.normalize_interval_array.repeat(bs, 1))
        # js_result = torch.add(js_result, self.normalize_center_array.repeat(bs, 1))

        return js_result

    @torch.no_grad()
    def forward_kinematics(self, joint_states: np.ndarray):
        joint_states = torch.Tensor(joint_states).to(self.device)
        bs = len(joint_states)
        pose_for_js = self.chain.forward_kinematics(joint_states)
        forward_result = slice_fk_pose(pose_for_js, bs)
        return forward_result.detach().cpu().numpy()

    @staticmethod
    def generate_cubic_bezier_trajectory(start_pose=None, target_pose=None, control_points=[], points=100):
        t_step = np.divide(1, points)
        points_set = np.empty((2 + len(control_points), 3))
        points_set[0] = start_pose[:3]
        for e, control_point in enumerate(control_points):
            #print(control_point[e])
            points_set[e + 1] = control_point[:3]
        points_set[-1] = target_pose[:3]
        t_points = np.arange(0, 1, t_step)
        t_points = np.concatenate((t_points, np.array([1.0])), axis=0)

        print(f' t_point: {t_points}')

        #print(points_set)
        curve = Bezier.Curve(t_points, points_set)
        #print(curve)

        diff_start_end_rotation = np.abs(np.subtract(target_pose[3:], start_pose[3:]))
        print(diff_start_end_rotation)
        end_diff = False
        for quat_val in diff_start_end_rotation:
            if quat_val > 0.03:
                end_diff = True
        
        print(f'orientation diff?: {end_diff}')

        #positive_difference = np.sum(np.abs(np.subtract(target_pose[3:], start_pose[3:])), axis=0)
        #negative_difference = np.sum(np.abs(np.subtract(-target_pose[3:], start_pose[3:])), axis=0)

        #if positive_difference > negative_difference:
        #    target_pose[3:] = -target_pose[3:]

        if control_points.shape[1] == 7 or (end_diff):
            key_rots = []
            key_rots.append(start_pose[3:])
            if control_points.shape[1] == 7:
                for control_rot in control_points:
                    key_rots.append(control_rot[3:])
            key_rots.append(target_pose[3:])

            print(f'ctrl points shape: {control_points.shape}')
            print(f'key rots: {key_rots}')

            #key_rots = np.array(key_rots)

            if control_points.shape[1] == 7:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n')
                key_times = list(np.arange(0, 1, (1 / (len(key_rots) - 1))))
                key_times.append(1)
                print(key_times)
            else:
                key_times = [0, 1]

            key_rots = R.from_quat(key_rots)

            slerp = Slerp(key_times, key_rots)

            interp_rots = slerp(list(t_points))
            interp_rots = np.array(list(interp_rots.as_quat()))
        else:
            fixed_rot = np.zeros((1, 4))
            fixed_rot[0] = target_pose[3:]
            interp_rots = np.repeat(fixed_rot, len(curve), axis=0)

        #print(interp_rots)
        res_curve = np.concatenate((np.array(curve), np.array(interp_rots)), axis=1)
        print(res_curve)
        return res_curve
        

