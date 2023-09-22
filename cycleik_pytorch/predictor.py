# Copyright (c) 2023. Jan-Gerrit Habekost. GNU General Public License https://www.gnu.org/licenses/gpl-3.0.html.

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
import pybullet as p
import concurrent.futures

class CycleIK:

    def __init__(self, robot, cuda_device=None, verbose=False):
        assert robot is not None and type(robot) == str, "Robot name must be set as string with valid config in path"
        self.robot = robot
        self.device_name = "cpu" if cuda_device is None else f"cuda:{cuda_device}"
        self.device = torch.device(self.device_name)
        self.config = load_config(robot)
        self.verbose = verbose

        self.robot_dof   = self.config["robot_dof"]
        self.robot_urdf  = self.config["robot_urdf"]
        self.robot_eef   = self.config["robot_eef"]

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

        self.gan = GenericNoisyGenerator(input_size=7, output_size=self.robot_dof,
                                      noise_vector_size=self.config["GAN"]["architecture"]["noise_vector_size"],
                                      nbr_tanh=self.config["GAN"]["architecture"]["nbr_tanh"],
                                      activation=self.config["GAN"]["architecture"]["activation"],
                                      layers=self.config["GAN"]["architecture"]["layers"]).to(self.device)

        data_path = str(Path(__file__).parent.parent.absolute())
        try:
            self.model.load_state_dict(torch.load(os.path.join(data_path, "weights", str(self.robot), "netG_B2A_with_kinematics.pth")))
        except FileNotFoundError:
            pass
        except RuntimeError:
            pass
        try:
            self.gan.load_state_dict(torch.load(os.path.join(data_path, "weights", str(self.robot), "netG_B2A_GAN_with_kinematics.pth")))
        except FileNotFoundError:
            pass
        except RuntimeError:
            pass
        self.model.eval()

        urdf_path = os.path.join(data_path, self.robot_urdf)
        chain = pk.build_serial_chain_from_urdf(open(urdf_path).read(), self.robot_eef)
        self.chain = chain.to(dtype=torch.float32, device=self.device)

    def __del__(self):
        p.disconnect()

    def check_single_collision(self, sample_js):
        for i, joint_id in enumerate(self.pybullet_joint_ids):
            # print(p.getJointInfo(self.pybullet_robot, jointIndex=joint_id))
            p.resetJointState(self.pybullet_robot, joint_id, self.collision_check_initial_position[i])

        # print(len(p.getContactPoints(self.pybullet_robot, self.pybullet_robot)))
        for i in range(1):
            p.stepSimulation(self.pybullet_robot)
        # time.sleep(0.001)

        initial_contacts = len(p.getContactPoints(self.pybullet_robot, self.pybullet_robot))
        # print(f"Initial contacts : {initial_contacts}")

        # time.sleep(15)

        res = p.setJointMotorControlArray(self.pybullet_robot, self.pybullet_joint_ids, controlMode=p.POSITION_CONTROL,
                                          targetPositions=sample_js)

        for i in range(1):
            p.stepSimulation(self.pybullet_robot)

        print("here")

        if len(p.getContactPoints(self.pybullet_robot, self.pybullet_robot)) > initial_contacts:
            return True
        else:
            return False

    def check_collision(self, joint_states):

        collision_counter = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # map
            l = list(executor.map(lambda sample_js: self.check_single_collision(sample_js), list(joint_states)))

        print(len(l))
        #for k in tqdm(range(len(joint_states))):
        #    sample_js = list(joint_states[k])
        #
        #
        #    if self.check_single_collision(sample_js):
        #        #print("self collision")
        #        collision_counter += 1

        print(collision_counter)

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
            error = torch.abs(torch.subtract(forward_result, unnormalized_poses)).detach().cpu().numpy()

        js_result = js_result.detach().cpu().numpy()

        if self.verbose:
            print(f'Execution time IK: {elapsed_time_ik} seconds')
            if calculate_error:
                print(f'Execution time FK: {elapsed_time_fk} seconds')
                print(f'Average FK Error: {error}')

        return js_result, error, elapsed_time_ik, elapsed_time_fk

    @torch.no_grad()
    def inverse_kinematics_distribution(self, pose: np.ndarray, js_samples = 1000, calculate_error=False):
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

        error = None
        elapsed_time_fk = None
        if calculate_error:
            st = time.time()
            pose_for_js = self.chain.forward_kinematics(js_result)
            forward_result = slice_fk_pose(pose_for_js, batch_size=js_samples)

            et = time.time()
            elapsed_time_fk = et - st
            print(f'PK Pose: {forward_result}')
            error = torch.mean(torch.abs(torch.subtract(forward_result, unnormalized_pose.repeat((js_samples, 1)))), dim=0).detach().cpu().numpy()

        js_result = js_result.detach().cpu().numpy()

        if self.verbose:
            print(f'Execution time IK: {elapsed_time_ik} seconds')
            if calculate_error:
                print(f'Execution time FK: {elapsed_time_fk} seconds')
                print(f'Average FK Error: {error}')

        return js_result, error, elapsed_time_ik, elapsed_time_fk

    @torch.no_grad()
    def forward_kinematics(self, joint_states: np.ndarray):
        joint_states = torch.Tensor(joint_states).to(self.device)
        bs = len(joint_states)
        pose_for_js = self.chain.forward_kinematics(joint_states)
        forward_result = slice_fk_pose(pose_for_js, bs)
        return forward_result.detach().cpu().numpy()
