import torch.utils.data
import os
import random
import pickle
import torch.backends.cudnn as cudnn
import torch.utils.data
from tqdm import tqdm
import optuna
from cycleik_pytorch import DecayLR, IKDataset, GenericGenerator, GenericDiscriminator, FineTuneModel
from cycleik_pytorch import weights_init, load_config, slice_fk_pose, normalize_pose, renormalize_pose, renormalize_joint_state
import pytorch_kinematics as pk
from .train import BaseTrainer


class FineTuneMLPTrainer(BaseTrainer):

    def __init__(self, args, trial=None, config=None, train_dataset=None, test_dataset=None):
        self.fine_tuning_model = None

        super().__init__(args, trial, config, train_dataset, test_dataset)

        self.position_losses = []
        self.fk_position_losses = []
        self.orientation_losses = []
        self.fk_orientation_losses = []
        self.var_losses = []
        self.val_position_losses = []
        self.val_fk_position_losses = []
        self.val_orientation_losses = []
        self.val_fk_orientation_losses = []
        self.val_varience_losses = []
        
        self.position_loss = 0
        self.orientation_loss = 0
        self.fk_position_loss = 0
        self.fk_orientation_loss = 0
        self.variance_loss = 0
        self.average_loss = 0.

        self.val_position_loss = 0
        self.val_orientation_loss = 0
        self.val_fk_position_loss = 0
        self.val_fk_orientation_loss = 0
        self.val_variance_loss = 0

        # define loss function (adversarial_loss) and optimizer
        # torch.nn.MSELoss().to(device)
        self.zero_joints_loss = torch.nn.MSELoss().to(self.device)

        try:
            self.robot_zero_joints = self.config["zero_joints_goal"]
        except KeyError:
            self.robot_zero_joints = None
            pass

        self.zero_joints_goal = True if self.robot_zero_joints is not None and len(self.robot_zero_joints) > 0 else False

        if self.zero_joints_goal:
            self.zero_joints_array = torch.zeros((self.batch_size, len(self.robot_zero_joints))).to(self.device)
            if self.train_data_size % self.batch_size != 0:
                self.small_zero_joints_array = torch.zeros((self.train_data_size % self.batch_size,
                                                           len(self.robot_zero_joints))).to(self.device)



    def calculate_zero_joints_loss(self, joint_state, batch_size, val=False):
        zero_joints_loss_ik = None
        joint_index_view = None
        for joint_index in self.robot_zero_joints:
            if joint_index_view is None:
                joint_index_view = joint_state.select(dim=1, index=joint_index)
                joint_index_view = joint_index_view.reshape(shape=(len(joint_index_view), 1))
            else:
                joint_index_view = torch.concat((joint_index_view, joint_state.select(dim=1, index=joint_index).reshape(
                    shape=(len(joint_index_view), 1))), dim=-1)
        if not val:
            if batch_size != self.batch_size:
                zero_joints_loss_ik = self.zero_joints_loss(joint_index_view, self.small_zero_joints_array)
            else:
                zero_joints_loss_ik = self.zero_joints_loss(joint_index_view, self.zero_joints_array)
        else:
            zero_joints_loss_ik = self.zero_joints_loss(joint_index_view, torch.zeros(len(joint_state), len(joint_index_view[0])).to(self.device))
        return zero_joints_loss_ik

    def training_step(self, data, i):
        gt_A = data["gt_A"].to(self.device)
        gt_B = data["gt_B"].to(self.device)
        real_B = data["real_B"].to(self.device)
        bs = len(gt_B)

        cycle_loss_A2B_position = None
        cycle_loss_A2B_orientation = None
        if self.autoencoder and not self.two_stage:
            fk_tensor = self.fk_model(gt_A)
            #print(fk_tensor[0])
            #forward_result = renormalize_pose(fk_tensor, bs, workspace_move=self.workspace_move,
            #                                  workspace_renormalize=self.workspace_renormalize)
            cycle_loss_A2B_position = self.cycle_loss(fk_tensor[:, :3], gt_B[:, :3])
            cycle_loss_A2B_orientation = self.cycle_loss(fk_tensor[:, 3:], gt_B[:, 3:])
            errD = cycle_loss_A2B_position * 3000 + cycle_loss_A2B_orientation * 3000
            errD.backward()

        backward_B2A = self.model(gt_B)

        cycle_loss_B2A_position = None
        cycle_loss_B2A_orientation = None
        if self.autoencoder:
            fk_tensor = self.fk_model(backward_B2A)
            #forward_result = renormalize_pose(fk_tensor, bs, workspace_move=self.workspace_move,
            #                                  workspace_renormalize=self.workspace_renormalize)
            cycle_loss_B2A_position = self.cycle_loss(fk_tensor[:, :3], gt_B[:, :3])
            cycle_loss_B2A_orientation = self.cycle_loss(fk_tensor[:, 3:], gt_B[:, 3:])
        else:
            js = renormalize_joint_state(backward_B2A, bs, single_renormalize=self.single_renormalize,
                                         single_renormalize_move=self.single_renormalize_move)
            fk_tensor = self.chain.forward_kinematics(js)
            forward_result = slice_fk_pose(fk_tensor, bs)
            cycle_loss_B2A_position = self.cycle_loss(forward_result[:, :3], real_B[:, :3])
            cycle_loss_B2A_orientation = self.cycle_loss(forward_result[:, 3:], real_B[:, 3:])

        zero_joints_loss_ik = None
        if self.zero_joints_goal:
            zero_joints_loss_ik = self.calculate_zero_joints_loss(backward_B2A, bs)
            errG = cycle_loss_B2A_position * 2000 + cycle_loss_B2A_orientation * 2000 + zero_joints_loss_ik * 2
        else:
            errG = cycle_loss_B2A_position * 500 + cycle_loss_B2A_orientation * 500# cycle_loss_B2A * 1000 + variance_loss_B2A#* 1000 + zero_joints_loss_ik
        errG.backward()


        return cycle_loss_B2A_position, cycle_loss_B2A_orientation, zero_joints_loss_ik, cycle_loss_A2B_position, cycle_loss_A2B_orientation

    def validation_step(self, data):
        gt_A = data["gt_A"].to(self.device)
        gt_B = data["gt_B"].to(self.device)
        real_B = data["real_B"].to(self.device)

        with torch.no_grad():
            bs = len(gt_B)

            cycle_loss_A2B_position = None
            cycle_loss_A2B_orientation = None
            if self.autoencoder and not self.two_stage:
                fk_tensor = self.fk_model(gt_A)
                # print(fk_tensor[0])
                forward_result = renormalize_pose(fk_tensor, bs, workspace_move=self.workspace_move,
                                                 workspace_renormalize=self.workspace_renormalize)
                cycle_loss_A2B_position = self.cycle_loss(forward_result[:, :3], real_B[:, :3])
                cycle_loss_A2B_orientation = self.cycle_loss(forward_result[:, 3:], real_B[:, 3:])

            backward_B2A = self.model(gt_B)

            if self.autoencoder:
                fk_tensor = self.fk_model(backward_B2A)
                forward_result = renormalize_pose(fk_tensor, bs, workspace_move=self.workspace_move,
                                                  workspace_renormalize=self.workspace_renormalize)
            else:
                js = renormalize_joint_state(backward_B2A, bs, single_renormalize=self.single_renormalize,
                                                               single_renormalize_move=self.single_renormalize_move)
                fk_tensor = self.chain.forward_kinematics(js)
                forward_result = slice_fk_pose(fk_tensor, bs)

            zero_joints_loss_ik = None
            if self.zero_joints_goal:
                zero_joints_loss_ik = self.calculate_zero_joints_loss(backward_B2A, bs, val=True)
            cycle_loss_B2A_position = self.cycle_loss(forward_result[:, :3], real_B[:, :3])
            cycle_loss_B2A_orientation = self.cycle_loss(forward_result[:, 3:], real_B[:, 3:])

            return cycle_loss_B2A_position, cycle_loss_B2A_orientation, zero_joints_loss_ik, cycle_loss_A2B_position, cycle_loss_A2B_orientation

    def create_model(self):
        super().create_model()

        self.model = GenericGenerator(input_size=7, output_size=self.core_model_config["robot_dof"],
                                                  nbr_tanh=self.core_model_config["IKNet"]["architecture"]["nbr_tanh"],
                                                  activation=self.core_model_config["IKNet"]["architecture"]["activation"],
                                                  layers=self.core_model_config["IKNet"]["architecture"]["layers"]).to(self.device)

        self.model.load_state_dict(
            torch.load(os.path.join("weights", self.core_model, f"model_MLP_with_kinematics.pth"),
                       map_location=self.device))

        self.model = FineTuneModel(self.model, output_dimension=self.robot_dof).to(self.device)

    def process_training_losses(self, cycle_loss_B2A_position, cycle_loss_B2A_orientation, variance_loss_B2A,
                                cycle_loss_A2B_position, cycle_loss_A2B_orientation, bs, epoch):
        position_error = cycle_loss_B2A_position.item()
        orientation_error = cycle_loss_B2A_orientation.item()
        if self.zero_joints_goal:
            variance_error = variance_loss_B2A.item()

        if self.autoencoder and not self.two_stage:
            fk_position_loss = cycle_loss_A2B_position.item()
            fk_orientation_loss = cycle_loss_A2B_orientation.item()
            self.fk_position_loss += fk_position_loss * (bs / self.batch_size)
            self.fk_orientation_loss += fk_orientation_loss * (bs / self.batch_size)

        loss_for_lr = (position_error + orientation_error) / 2
        self.average_loss += loss_for_lr * (bs / self.batch_size)
        self.position_loss += position_error * (bs / self.batch_size)
        self.orientation_loss += orientation_error * (bs / self.batch_size)
        if self.zero_joints_goal:
            self.variance_loss += variance_error * (bs / self.batch_size)

        if self.autoencoder and not self.two_stage:
            self.progress_bar.set_description(
                f"[{epoch}/{self.epochs - 1}][{epoch}/{len(self.train_dataloader) - 1}] "
                f"fk_position_loss:: {fk_position_loss:.8f} "
                f"fk_orientation_loss:: {fk_orientation_loss:.8f} "
                f"position_loss:: {position_error:.8f} "
                f"orientation_loss:: {orientation_error:.8f} "
                f"zero_controller_loss:: {variance_error:.8f} "
                #    f"cycle_loss_var: {cycle_loss_var.item():.8f} "
            )
        else:
            if self.zero_joints_goal:
                self.progress_bar.set_description(
                    f"[{epoch}/{self.epochs - 1}][{epoch}/{len(self.train_dataloader) - 1}] "
                    f"position_loss:: {position_error:.8f} "
                    f"orientation_loss:: {orientation_error:.8f} "
                    f"zero_controller_loss:: {variance_error:.8f} "
                    #    f"cycle_loss_var: {cycle_loss_var.item():.8f} "
                )
            else:
                self.progress_bar.set_description(
                    f"[{epoch}/{self.epochs - 1}][{epoch}/{len(self.train_dataloader) - 1}] "
                    f"position_loss:: {position_error:.8f} "
                    f"orientation_loss:: {orientation_error:.8f} "
                    #f"zero_controller_loss:: {variance_error:.8f} "
                    #    f"cycle_loss_var: {cycle_loss_var.item():.8f} "
                )
    def process_validation_losses(self, cycle_loss_B2A_position, cycle_loss_B2A_orientation,
                                  variance_loss_B2A, cycle_loss_A2B_position, cycle_loss_A2B_orientation, bs, epoch):
        pos_loss = cycle_loss_B2A_position.item()
        rot_loss = cycle_loss_B2A_orientation.item()
        if self.zero_joints_goal:
            var_loss = variance_loss_B2A.item()

        if self.autoencoder and not self.two_stage:
            fk_position_loss = cycle_loss_A2B_position.item()
            fk_orientation_loss = cycle_loss_A2B_orientation.item()
            self.val_fk_position_loss += fk_position_loss * (bs / self.batch_size)
            self.val_fk_orientation_loss += fk_orientation_loss * (bs / self.batch_size)

        if self.zero_joints_goal:
            self.val_variance_loss += var_loss * (bs / 10000)
        self.val_position_loss += pos_loss * (bs / 10000)
        self.val_orientation_loss += rot_loss * (bs / 10000)

        # Set G_A and G_B's gradients to zero
        if self.autoencoder and not self.two_stage:
            self.progress_bar.set_description(
                f"[{epoch}/{self.epochs - 1}][{epoch}/{len(self.train_dataloader) - 1}] "
                f"fk_position_loss:: {fk_position_loss:.8f} "
                f"fk_orientation_loss:: {fk_orientation_loss:.8f} "
                f"position_loss:: {pos_loss:.8f} "
                f"orientation_loss:: {rot_loss:.8f} "
                f"zero_controller_loss:: {var_loss:.8f} "
                #    f"cycle_loss_var: {cycle_loss_var.item():.8f} "
            )
        else:
            if self.zero_joints_goal:
                self.progress_bar.set_description(
                    f"[{epoch}/{self.epochs - 1}][{epoch}/{len(self.train_dataloader) - 1}] "
                    f"position_loss:: {pos_loss:.8f} "
                    f"orientation_loss:: {rot_loss:.8f} "
                    f"zero_controller_loss:: {var_loss:.8f} "
                    #    f"cycle_loss_var: {cycle_loss_var.item():.8f} "
                )
            else:
                self.progress_bar.set_description(
                    f"[{epoch}/{self.epochs - 1}][{epoch}/{len(self.train_dataloader) - 1}] "
                    f"position_loss:: {pos_loss:.8f} "
                    f"orientation_loss:: {rot_loss:.8f} "
                    #f"zero_controller_loss:: {variance_error:.8f} "
                    #    f"cycle_loss_var: {cycle_loss_var.item():.8f} "
                )
    
    def create_lr_scheduler(self):
        super().create_lr_scheduler()
        lr_lambda = DecayLR(self.epochs, 0, self.decay_epochs).step
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda, verbose=True)

    def create_optimizer(self):
        super().create_optimizer()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

    def create_checkpoint(self, epoch):
        if epoch % 5 == 0 and not self.optimizer_run:
            torch.save(self.model.state_dict(), f"weights/{self.robot}/model_MLP_epoch_{epoch}_finetuned.pth")

    def train(self):
        torch.cuda.empty_cache()

        for epoch in range(0, self.epochs):
            self.progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
            self.position_loss = 0
            self.orientation_loss = 0
            self.variance_loss = 0
            self.average_loss = 0.
            self.fk_position_loss = 0
            self.fk_orientation_loss = 0

            for i, data in self.progress_bar:
                self.optimizer.zero_grad()
                if self.autoencoder and not self.two_stage:
                    self.fk_optimizer.zero_grad()
                cycle_loss_B2A_position, cycle_loss_B2A_orientation, variance_loss_B2A, \
                    cycle_loss_A2B_position, cycle_loss_A2B_orientation = self.training_step(data, i)
                self.optimizer.step()
                if self.autoencoder and not self.two_stage:
                    self.fk_optimizer.step()
                self.process_training_losses(cycle_loss_B2A_position, cycle_loss_B2A_orientation,
                                             variance_loss_B2A, cycle_loss_A2B_position,
                                             cycle_loss_A2B_orientation, len(data["gt_B"]), epoch)

            self.position_losses.append(self.position_loss)
            self.orientation_losses.append(self.orientation_loss)
            self.fk_position_losses.append(self.fk_position_loss)
            self.fk_orientation_losses.append(self.fk_orientation_loss)
            self.var_losses.append(self.variance_loss)

            self.create_checkpoint(epoch)

            self.val_position_loss = 0
            self.val_orientation_loss = 0
            self.val_variance_loss = 0
            self.val_fk_position_loss = 0
            self.val_fk_orientation_loss = 0

            if epoch == 0 or (epoch + 1) % 10 == 0:
                self.progress_bar = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader))
                for i, data in self.progress_bar:
                    # get batch size data
                    cycle_loss_B2A_val_position, cycle_loss_B2A_val_orientation, variance_loss_B2A, \
                        cycle_loss_A2B_position, cycle_loss_A2B_orientation = self.validation_step(data)
                    self.process_validation_losses(cycle_loss_B2A_val_position,
                                                   cycle_loss_B2A_val_orientation, variance_loss_B2A,
                                                   cycle_loss_A2B_position, cycle_loss_A2B_orientation,
                                                   len(data["gt_B"]), epoch)

            self.val_varience_losses.append(self.val_variance_loss)
            self.val_orientation_losses.append(self.val_orientation_loss)
            self.val_position_losses.append(self.val_position_loss)
            self.val_fk_position_losses.append(self.val_fk_position_loss)
            self.val_fk_orientation_losses.append(self.val_fk_orientation_loss)

            avg_loss = self.average_loss / (self.train_data_size / self.batch_size)

            if self.optimizer_run:
                self.trial.report(avg_loss, epoch)

                if self.trial.should_prune():
                    raise optuna.TrialPruned()

            print("Avg Loss: {0}".format(avg_loss))

            self.lr_scheduler.step()

        print([self.position_losses, self.orientation_losses, self.var_losses,
               self.val_position_losses, self.val_orientation_losses, self.val_varience_losses])

        if not self.optimizer_run:
            # save last check pointing
            torch.save(self.model.state_dict(), f"weights/{self.robot}/model_MLP_finetuned.pth")
            with open(rf"results/{self.robot}/train_MLP_finetuned_loss.p", "wb") as output_file:
                pickle.dump([self.position_losses, self.orientation_losses, self.var_losses,
               self.val_position_losses, self.val_orientation_losses, self.val_varience_losses], output_file)

        return ((self.val_position_losses[-1]) + self.val_orientation_losses[-1] / 2) + (self.val_varience_losses[-1] * 10)