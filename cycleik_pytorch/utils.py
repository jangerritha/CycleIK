import random
import torch
import yaml
import os
import numpy as np
from pathlib import Path
import pytorch_kinematics as pk

class ReplayBuffer:
    def __init__(self, max_size=10000):
        assert (max_size > 0), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        #print("unsqueeze: {0}".format(data))
        if len(self.data) < self.max_size:
            self.data.append(data)
            to_return.append(data)
        else:
            if random.uniform(0, 1) > 0.5:
                to_return.append(data)
                i = random.randint(0, self.max_size - 1)
                self.data[i] = data
            else:
                to_return.append(data)
        return torch.cat(to_return)

def load_config(robot):
    data_path = str(Path(__file__).parent.parent.absolute())
    with open(data_path + f"/config/{robot}.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

def get_kinematic_params(config):
    workspace_upper = np.array(config["workspace"]["upper"])
    workspace_lower = np.array(config["workspace"]["lower"])
    orig_workspace_upper = np.copy(workspace_upper)
    orig_workspace_lower = np.copy(workspace_lower)
    diff_upper_lower = np.abs(config["workspace"]["lower"]) - np.abs(config["workspace"]["upper"])
    workspace_interval_array = np.ones(3)
    workspace_center_array = np.zeros(3)
    assert len(diff_upper_lower) == 3 == len(config["workspace"]["lower"]) == len(config["workspace"]["upper"])

    for r in range(len(diff_upper_lower)):
        if diff_upper_lower[r] == 0 and workspace_upper[r] >= 0. >= workspace_lower[r]:
            workspace_interval_array[r] = abs(workspace_upper[r])
        elif diff_upper_lower[r] != 0 and workspace_upper[r] >= 0. >= workspace_lower[r]:
            half_diff = (diff_upper_lower[r] / 2)
            workspace_center_array[r] = half_diff
            workspace_interval_array[r] = max([abs(workspace_lower[r]), abs(workspace_upper[r])]) - abs(half_diff)
        elif diff_upper_lower[r] != 0 and ((workspace_upper[r] > 0. and workspace_lower[r] > 0.) or
                                           (workspace_upper[r] < 0. and workspace_lower[r] < 0.)):
            if workspace_upper[r] > 0. and workspace_lower[r] > 0.:
                workspace_center_array[r] = -workspace_lower[r]
                workspace_upper[r] = workspace_upper[r] - workspace_lower[r]
                workspace_lower[r] = 0.
                print(workspace_center_array[r])

            elif workspace_upper[r] < 0. and workspace_lower[r] < 0.:
                #print("Here")
                workspace_center_array[r] = abs(workspace_upper[r])
                workspace_lower[r] = workspace_lower[r] + abs(workspace_upper[r])
                workspace_upper[r] = 0.

            # local_diff_upper_lower = np.abs(config["limits"]["lower"]) - np.abs(config["limits"]["upper"])
            local_diff = workspace_lower[r] - workspace_upper[r]
            #print(local_diff)
            half_diff = (local_diff / 2)
            workspace_center_array[r] += half_diff
            print(workspace_center_array[r])

            workspace_interval_array[r] = abs(half_diff)

        else:
            raise NotImplementedError

    #workspace_interval_array = workspace_upper
    print(workspace_center_array)
    print(workspace_interval_array)

    limits_upper = np.array(config["limits"]["upper"])
    limits_lower = np.array(config["limits"]["lower"])
    orig_limits_upper = np.copy(limits_upper)
    orig_limits_lower = np.copy(limits_lower)
    diff_upper_lower = np.abs(config["limits"]["lower"]) - np.abs(config["limits"]["upper"])
    normalize_interval_array = np.ones(config["robot_dof"])
    normalize_center_array = np.zeros(config["robot_dof"])
    assert len(diff_upper_lower) == config["robot_dof"] == len(config["limits"]["lower"]) == len(config["limits"]["upper"])

    for r in range(len(diff_upper_lower)):
        if diff_upper_lower[r] == 0 and limits_upper[r] >= 0. >= limits_lower[r]:
            normalize_interval_array[r] = abs(limits_upper[r])
        elif diff_upper_lower[r] != 0 and limits_upper[r] >= 0. >= limits_lower[r]:
            half_diff = (diff_upper_lower[r] / 2)
            normalize_center_array[r] = half_diff
            normalize_interval_array[r] = max([abs(limits_lower[r]), abs(limits_upper[r])]) - abs(half_diff)
        elif diff_upper_lower[r] != 0 and ((limits_upper[r] > 0. and limits_lower[r] > 0.) or
                                           (limits_upper[r] < 0. and limits_lower[r] < 0.)):
            if limits_upper[r] > 0. and limits_lower[r] > 0.:
                normalize_center_array[r] = -limits_lower[r]
                limits_upper[r] = limits_upper[r] - limits_lower[r]
                limits_lower[r] = 0.
            elif limits_upper[r] < 0. and limits_lower[r] < 0.:
                print("Here")
                normalize_center_array[r] = abs(limits_upper[r])
                limits_lower[r] = limits_lower[r] + abs(limits_upper[r])
                limits_upper[r] = 0.

            # local_diff_upper_lower = np.abs(config["limits"]["lower"]) - np.abs(config["limits"]["upper"])
            local_diff = limits_lower[r] - limits_upper[r]
            print(local_diff)
            half_diff = -(local_diff / 2)
            normalize_center_array[r] += half_diff
            print(normalize_center_array[r])
            normalize_interval_array[r] = abs(half_diff)
            print(normalize_interval_array[r])
        else:
            raise NotImplementedError

    return workspace_interval_array, workspace_center_array, orig_limits_upper, orig_limits_lower, normalize_interval_array, normalize_center_array


def renormalize_joint_state(joint_state, batch_size, single_renormalize, single_renormalize_move):
    joint_state = torch.mul(joint_state, single_renormalize.repeat(batch_size, 1))
    joint_state = torch.subtract(joint_state, single_renormalize_move.repeat(batch_size, 1))
    return joint_state


def slice_fk_pose(pose, batch_size, rotation='quaternion'):
    pos = torch.reshape(pose.get_matrix()[:, :3, 3:], shape=(batch_size, 3))
    rot = None
    if rotation == 'quaternion':
        rot = pk.matrix_to_quaternion(pose.get_matrix()[:, :3, :3])
        rot = torch.concat((rot[:, 1:], rot[:, :1]), dim=1)
    elif rotation == 'angle':
        rot = pk.matrix_to_euler_angles(pose.get_matrix()[:, :3, :3], convention='ZYX')
    else:
        raise NotImplementedError
    return torch.concat((pos, rot), dim=1)


def normalize_pose(pose, batch_size, workspace_move, workspace_renormalize, slice_pk_result=True):
    if slice_pk_result:
        forward_result = slice_fk_pose(pose, batch_size)
    else:
        forward_result = pose
    forward_result[:, :3] = torch.add(forward_result[:, :3], workspace_move)
    forward_result[:, :3] = torch.true_divide(forward_result[:, :3], workspace_renormalize)
    return forward_result


def renormalize_pose(pose, batch_size, workspace_move, workspace_renormalize):
    pose[:, :3] = torch.mul(pose[:, :3], workspace_renormalize.repeat(batch_size, 1))
    pose[:, :3] = torch.subtract(pose[:, :3], workspace_move.repeat(batch_size, 1))
    return pose


class JSD(torch.nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))