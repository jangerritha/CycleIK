import torch.utils.data
from cycleik_pytorch import DecayLR, IKDataset, GenericNoisyGenerator, GenericDiscriminator
import os
import random
import pickle
import torch.backends.cudnn as cudnn
import torch.utils.data
from tqdm import tqdm
import optuna
from cycleik_pytorch import DecayLR, IKDataset, GenericGenerator
from cycleik_pytorch import weights_init, load_config, slice_fk_pose, normalize_pose, renormalize_pose, renormalize_joint_state
import pytorch_kinematics as pk
from .train import BaseTrainer


class FKTrainer(BaseTrainer):

    def __init__(self, args, trial=None, config=None, train_dataset=None, test_dataset=None):
        super().__init__(args, trial, config, train_dataset, test_dataset)

        self.position_losses = []
        self.orientation_losses = []
        self.val_position_losses = []
        self.val_orientation_losses = []

        self.train_position_loss = 0
        self.train_orientation_loss = 0
        self.val_position_loss = 0
        self.val_orientation_loss = 0

        self.average_loss = 0

        self.kinematic_loss = torch.nn.L1Loss().to(self.device)

    def training_step(self, data, i):
        gt_A = data["gt_A"].to(self.device)
        real_B = data["real_B"].to(self.device)

        forward_A2B = self.fk_model(gt_A)
        forward_result = renormalize_pose(forward_A2B.clone(), batch_size=len(forward_A2B),
                        workspace_renormalize=self.workspace_renormalize, workspace_move=self.workspace_move)
        position_error = self.kinematic_loss(forward_result[:, :3], real_B[:, :3])
        orientation_error = self.kinematic_loss(forward_result[:, 3:], real_B[:, 3:])
        errG = position_error + orientation_error
        errG.backward()

        return position_error, orientation_error

    def validation_step(self, data):
        gt_A = data["gt_A"].to(self.device)
        real_B = data["real_B"].to(self.device)

        with torch.no_grad():
            forward_A2B = self.fk_model(gt_A)
            forward_A2B = renormalize_pose(forward_A2B, batch_size=len(forward_A2B),
                                           workspace_renormalize=self.workspace_renormalize,
                                           workspace_move=self.workspace_move)
            cycle_loss_B2A_position = self.kinematic_loss(forward_A2B[:, :3], real_B[:, :3])
            cycle_loss_B2A_orientation = self.kinematic_loss(forward_A2B[:, 3:], real_B[:, 3:])
            self.val_position_loss += cycle_loss_B2A_position.item() * (len(real_B) / 10000)
            self.val_orientation_loss += cycle_loss_B2A_orientation.item() * (len(real_B) / 10000)

            return cycle_loss_B2A_position, cycle_loss_B2A_orientation

    def process_training_losses(self, cycle_loss_B2A_position, cycle_loss_B2A_orientation, bs, epoch):
        position_error = cycle_loss_B2A_position.item()
        orientation_error = cycle_loss_B2A_orientation.item()

        loss_for_lr = (position_error + orientation_error) / 2
        self.average_loss += loss_for_lr * (bs / self.batch_size)
        self.train_position_loss += position_error * (bs / self.batch_size)
        self.train_orientation_loss += orientation_error * (bs / self.batch_size)

        self.progress_bar.set_description(
            f"[{epoch}/{self.epochs - 1}][{epoch}/{len(self.train_dataloader) - 1}] "
            f"position_loss: {cycle_loss_B2A_position.item():.8f} "
            f"orientation_loss: {cycle_loss_B2A_orientation.item():.8f} ")

    def process_validation_losses(self, cycle_loss_B2A_position, cycle_loss_B2A_orientation, bs, epoch):
        pos_loss = cycle_loss_B2A_position.item()
        rot_loss = cycle_loss_B2A_orientation.item()

        self.val_position_loss += pos_loss * (bs / 10000)
        self.val_orientation_loss += rot_loss * (bs / 10000)

        # Set G_A and G_B's gradients to zero

        self.progress_bar.set_description(
            f"[{epoch}/{self.epochs - 1}][{epoch}/{len(self.test_dataloader) - 1}] "
            f"position_loss:: {pos_loss:.8f} "
            f"orientation_loss:: {rot_loss:.8f} "
        )

    def create_checkpoint(self, epoch):
        if epoch % 5 == 0 and not self.optimizer_run:
            torch.save(self.fk_model.state_dict(), f"weights/{self.robot}/model_FK_epoch_{epoch}_with_kinematics.pth")

    def train(self):
        torch.cuda.empty_cache()

        for epoch in range(0, self.epochs):
            self.progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
            self.train_position_loss = 0
            self.train_orientation_loss = 0
            self.average_loss = 0

            for i, data in self.progress_bar:
                self.fk_optimizer.zero_grad()
                cycle_loss_B2A_position, cycle_loss_B2A_orientation = self.training_step(data, i)
                self.fk_optimizer.step()
                self.process_training_losses(cycle_loss_B2A_position, cycle_loss_B2A_orientation,
                                             len(data["gt_B"]), epoch)

            self.position_losses.append(self.train_position_loss)
            self.orientation_losses.append(self.train_orientation_loss)

            self.create_checkpoint(epoch)

            self.val_position_loss = 0
            self.val_orientation_loss = 0

            if epoch == 0 or (epoch + 1) % 10 == 0:
                self.progress_bar = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader))
                for i, data in self.progress_bar:
                    # get batch size data
                    cycle_loss_B2A_val_position, cycle_loss_B2A_val_orientation = self.validation_step(data)
                    self.process_validation_losses(cycle_loss_B2A_val_position, cycle_loss_B2A_val_orientation,
                                                   len(data["gt_B"]), epoch)

            self.val_orientation_losses.append(self.val_orientation_loss)
            self.val_position_losses.append(self.val_position_loss)

            avg_loss = self.average_loss / (self.train_data_size / self.batch_size)

            if self.optimizer_run:
                self.trial.report(avg_loss, epoch)

                if self.trial.should_prune():
                    raise optuna.TrialPruned()

            print("Avg Loss: {0}".format(avg_loss))

            self.fk_lr_scheduler.step()

        print([self.position_losses, self.orientation_losses, self.val_position_losses, self.val_orientation_losses])

        if not self.optimizer_run:
            # save last check pointing
            torch.save(self.fk_model.state_dict(), f"weights/{self.robot}/model_FK_with_kinematics.pth")
            with open(rf"results/{self.robot}/train_FK_with_kinematics_loss.p", "wb") as output_file:
                pickle.dump([self.position_losses, self.orientation_losses,
                self.val_position_losses, self.val_orientation_losses], output_file)

        return (self.val_position_losses[-1]) + self.val_orientation_losses[-1] / 2