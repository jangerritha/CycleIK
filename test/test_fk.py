from .test import BaseTester
import numpy as np
import torch
from cycleik_pytorch import load_config, renormalize_pose, normalize_pose, slice_fk_pose, renormalize_joint_state, IKDataset
from cycleik_pytorch import GenericGenerator, GenericDiscriminator
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


class FKTester(BaseTester):

    def __init__(self, args):
        super().__init__(args)
        self.error_list = []
        self.min_error = 10000
        self.max_error = 0
        self.avg_error = 0

        self.min_error_ik = 10000
        self.max_error_ik = 0
        self.avg_ik_error = [0., 0., 0., 0., 0., 0.]

        self.over_1mm = 0
        self.over_2_5mm = 0
        self.over_5mm = 0
        self.over_1cm = 0
        self.over_1_5cm = 0
        self.over_2cm = 0
        self.over_5cm = 0
        self.over_10cm = 0

        self.over_1mm_ik = 0
        self.over_2_5mm_ik = 0
        self.over_5mm_ik = 0
        self.over_1cm_ik = 0
        self.over_1_5cm_ik = 0
        self.over_2cm_ik = 0
        self.over_5cm_ik = 0
        self.over_10cm_ik = 0

        self.it_counter = 0

    def test_step(self, **kwargs):
        pass

    def process_test_losses(self, **kwargs):
        pass

    def create_model(self):
        self.model = GenericDiscriminator(input_size=self.robot_dof, output_size=7,
                                      nbr_tanh=self.config["FKNet"]["architecture"]["nbr_tanh"],
                                      activation=self.config["FKNet"]["architecture"]["activation"],
                                      layers=self.config["FKNet"]["architecture"]["layers"]).to(self.device)

    def load_dataset(self):
        self.dataset = IKDataset(root=self.val_data, test=True, robot=self.robot, config=self.config)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=10000, shuffle=False, pin_memory=True)

    def test(self):
        avg_ik_error = [0., 0., 0., 0., 0., 0.]
        for i, data in self.progress_bar:  # for i, data in progress_bar
            # get batch size data
            gt_A = data["gt_A"].to(self.device)
            gt_B = data["gt_B"].to(self.device)
            real_B = data["real_B"].to(self.device)
            # print(real_images_B.cpu().detach().numpy().shape)
            # print("-----------------------------------------------")

            # print("----- IK test -------")
            with torch.no_grad():
                result_B = self.model(gt_A)

            # ik_error = np.average(abs(moveit_fk_array - gt_b_np), axis=0)

            #js = renormalize_joint_state(result_B, batch_size=len(gt_B),
            #                             single_renormalize_move=self.single_renormalize_move,
            #                             single_renormalize=self.single_renormalize)
            #fk_tensor = self.chain.forward_kinematics(js)
            #forward_result = slice_fk_pose(fk_tensor, batch_size=len(gt_B))

            gt_b_np = gt_B.detach().cpu().numpy()
            real_b_np = real_B.detach().cpu().numpy()
            forward_result = renormalize_pose(result_B, len(gt_A), workspace_move=self.workspace_renormalize_move,
                                              workspace_renormalize=self.workspace_renormalize)
            forward_result_np = forward_result.clone().detach().cpu().numpy()

            r_target = R.from_quat(list(gt_b_np[:, 3:]))
            r_result = R.from_quat(list(forward_result_np[:, 3:]))
            # print(r_result)
            rotation_diff = r_target * r_result.inv()
            rotation_diff_save = rotation_diff.as_euler('zyx', degrees=True)
            rotation_diff = np.abs(rotation_diff_save)
            # print(rotation_diff_save.shape)
            # print(abs(result_A_np[:,:3] - gt_b_np[:,:3]))
            # print(np.array(rotation_diff))

            # print(np.abs(np.multiply(forward_result[:, :3], single_normalize_pos_np) - np.multiply(gt_b_np[:, :3], single_normalize_pos_np)))
            # print(np.array(rotation_diff))
            ik_error_batch = np.concatenate(
                (np.reshape(np.abs(forward_result_np[:, :3] - real_b_np[:, :3]), newshape=(10000, 3)),
                 np.array(rotation_diff)), axis=1)

            # print(ik_error)
            # print("FK Moveit: ", moveit_fk_array)

            # print(error)
            for k, ik_error in enumerate(ik_error_batch):
                if np.sum(ik_error[:3]) / 3 < 0.01 and np.sum(ik_error[3:]) / 3 < 20.:
                    self.count_success += 1
                if ik_error[0] == 0. and ik_error[1] == 0. and ik_error[2] == 0.:
                    print(k)
                    print("--------------------")

                self.error_list.append(ik_error)
                avg_ik_error = np.add(avg_ik_error, ik_error)

                # ik_error[:3] = np.true_divide(ik_error[:3], 1000)
                ik_error = np.average(ik_error[:3])

                if self.min_error_ik > ik_error:
                    self.min_error_ik = ik_error
                if self.max_error_ik < ik_error:
                    self.max_error_ik = ik_error

                if ik_error > 0.001:
                    self.over_1mm_ik += 1
                if ik_error > 0.0025:
                    self.over_2_5mm_ik += 1
                if ik_error > 0.005:
                    self.over_5mm_ik += 1
                if ik_error > 0.01:
                    self.over_1cm_ik += 1
                if ik_error > 0.015:
                    self.over_1_5cm_ik += 1
                if ik_error > 0.02:
                    self.over_2cm_ik += 1
                if ik_error > 0.05:
                    self.over_5cm_ik += 1
                if ik_error > 0.1:
                    self.over_10cm_ik += 1

                # rospy.sleep(0.00001)

        avg_ik_error = np.true_divide(avg_ik_error, self.samples)
        avg_ik_error_mm = np.copy(avg_ik_error)
        avg_ik_error_mm[:3] = np.multiply(avg_ik_error_mm[:3], 1000)

        error_array = np.array(self.error_list, dtype=np.float32)

        error_array_mean = np.concatenate((
            np.reshape(np.average(error_array[:, :3], axis=1), newshape=(self.samples, 1)),
            np.reshape(np.average(error_array[:, 3:], axis=1), newshape=(self.samples, 1))
        ), axis=1)

        error_array_mean_cleaned = np.copy(error_array_mean)

        outlier_rows = []
        for r, row in enumerate(error_array_mean):
            if row[0] > 0.01 or row[1] > 1.:
                outlier_rows.append(r)

        error_array_mean_cleaned = np.delete(error_array_mean_cleaned, outlier_rows, axis=0)

        print(len(error_array_mean_cleaned))

        max_pos_error_all_axis, max_rot_error_all_axis = np.max(error_array[:, :3], axis=0), np.max(error_array[:, 3:],
                                                                                                    axis=0)
        min_pos_error_all_axis, min_rot_error_all_axis = np.min(error_array[:, :3], axis=0), np.min(error_array[:, 3:],
                                                                                                    axis=0)
        max_pos_err_all_axis_mm = np.multiply(max_pos_error_all_axis, 1000)
        min_pos_err_all_axis_mm = np.multiply(min_pos_error_all_axis, 1000)

        max_pos_err, max_rot_error = np.max(error_array_mean, axis=0)[0], np.max(error_array_mean, axis=0)[1]
        min_pos_err, min_rot_error = np.min(error_array_mean, axis=0)[0], np.min(error_array_mean, axis=0)[1]
        max_pos_err_mm = np.multiply(max_pos_err, 1000)
        min_pos_err_mm = np.multiply(min_pos_err, 1000)
        mean_pos_err = np.average(avg_ik_error_mm[:3])
        mean_rot_err = np.average(avg_ik_error_mm[3:])

        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)
        # progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        measure_time = 0.
        # for i, data in progress_bar:#for i, data in progress_bar
        #    # get batch size data
        #    gt_A = data["gt_A"].to(device)
        #    gt_B = data["gt_B"].to(device)
        #
        #
        #
        #    with torch.no_grad():
        #        start_time = time.time()
        #        result_B = netG_B2A(gt_B)
        #        end_time = time.time()
        #        measure_time += end_time - start_time

        print(f'\nAverage IK error over all axis: XYZ (mm): {avg_ik_error_mm[:3]}, RPY: {avg_ik_error_mm[3:]}')
        print(f'Max IK error over all axis: XYZ (mm): {max_pos_err_all_axis_mm}, RPY: {max_rot_error_all_axis}')
        print(f'Min IK error over all axis: XYZ (mm): {min_pos_err_all_axis_mm}, RPY: {min_rot_error_all_axis}\n')

        print(f'\nAverage IK error over all axis: XYZ (mm): {mean_pos_err}, RPY: {mean_rot_err}')
        print(f'Max IK error : XYZ (mm): {max_pos_err_mm}, RPY: {max_rot_error}')
        print(f'Min IK error: XYZ (mm): {min_pos_err_mm}, RPY: {min_rot_error}\n')

        print(f"Avg inference_time: {measure_time / self.samples}")
        print(f"Success Rate: {self.count_success / self.samples}")

        print("--------- IK Results -------------")
        # print("Average IK Error: ", avg_ik_error[0,:3])
        print("Average IK Error: \n"
              "     Position (mm): {0}".format(avg_ik_error[:3]) + "\n"
                                                                   "     Orientation (degree): {0}".format(
            avg_ik_error[3:]))
        print("Min Error: " + str(self.min_error_ik))
        print("Max Error: " + str(self.max_error_ik))
        print("Error > 1mm: ", self.over_1mm_ik)
        print("Percent > 1mm: ", self.over_1mm_ik / self.samples)
        print("Error > 2.5mm: ", self.over_2_5mm_ik)
        print("Percent > 2.5mm: ", self.over_2_5mm_ik / self.samples)
        print("Error > 5mm: ", self.over_5mm_ik)
        print("Percent > 5mm: ", self.over_5mm_ik / self.samples)
        print("Error > 1cm: ", self.over_1cm_ik)
        print("Percent > 1cm: ", self.over_1cm_ik / self.samples)
        print("Error > 1.5cm: ", self.over_1_5cm_ik)
        print("Percent > 1.5cm: ", self.over_1_5cm_ik / self.samples)
        print("Error > 2cm: ", self.over_2cm_ik)
        print("Percent > 2cm: ", self.over_2cm_ik / self.samples)
        print("Error > 5cm: ", self.over_5cm_ik)
        print("Percent > 5cm: ", self.over_5cm_ik / self.samples)
        print("Error > 10cm: ", self.over_10cm_ik)
        print("Percent > 10cm: ", self.over_10cm_ik / self.samples)
        # print("Average IK Error: " + str(ik_error))

        bins = [0.0]
        for i in range(240):
            bins.append(bins[-1] + 0.005)

        rot_bins = [0.0]
        for i in range(320):
            rot_bins.append(rot_bins[-1] + 0.0005)

        fig, ax = plt.subplots(2)

        ax[0].set_xlabel('Error (mm)')
        ax[0].set_ylabel('Count')
        n, bins, patches = ax[0].hist(np.multiply(error_array_mean[:, :1], 1000), bins=bins)
        mean_error = ax[0].axvline(np.mean(np.multiply(error_array_mean[:, :1], 1000), axis=0), color='orange',
                                   linestyle='dashed', linewidth=1)
        mean_error.set_label('Mean Absolute Error (all solutions)')
        mean_error = ax[0].axvline(np.mean(np.multiply(error_array_mean_cleaned[:, :1], 1000), axis=0), color='green',
                                   linestyle='dashed', linewidth=1)
        mean_error.set_label('Mean Absolute Error (only valid solutions)')
        bins = bins + 0.0025
        ax[0].plot(bins[:240], n, linewidth=1, linestyle=(0, (5, 1)), color="black")
        ax[0].set_xticks(np.arange(0.0, 1.4, 0.2))
        ax[0].set_yticks(np.arange(0.0, 7000., 1000.))
        ax[0].xaxis.label.set_fontweight('bold')
        ax[0].yaxis.label.set_fontweight('bold')
        ax[0].legend()

        ax[1].set_xlabel('Error (degree)')
        ax[1].set_ylabel('Count')
        n, bins, patches = ax[1].hist(error_array_mean[:, 1:], bins=rot_bins)
        mean_error = ax[1].axvline(error_array_mean[:, 1:].mean(), color='orange', linestyle='dashed', linewidth=1)
        mean_error.set_label('Mean Absolute Error (all solutions)')
        mean_error = ax[1].axvline(error_array_mean_cleaned[:, 1:].mean(), color='green', linestyle='dashed',
                                   linewidth=1)
        mean_error.set_label('Mean Absolute Error (only valid solutions)')
        bins = bins + 0.00025
        ax[1].plot(bins[:320], n, linewidth=1, linestyle=(0, (5, 1)), color="black")
        ax[1].set_xticks(np.arange(0.0, 0.18, 0.02))
        # ax[1].set_xlim(right=0.16)
        ax[1].set_yticks(np.arange(0.0, 5000., 1000.))
        ax[1].xaxis.label.set_fontweight('bold')
        ax[1].yaxis.label.set_fontweight('bold')
        ax[1].legend()

        fig.tight_layout()
        plt.savefig('./img/losses/vis_error_dist_iknet.png')
        plt.show()