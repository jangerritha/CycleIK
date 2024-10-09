import numpy as np
import torch.utils.data
import os
import random
import pickle
import torch.backends.cudnn as cudnn
import torch.utils.data
from tqdm import tqdm
import optuna
from cycleik_pytorch import DecayLR, IKDataset, GenericGenerator, GenericDiscriminator
from cycleik_pytorch import weights_init, load_config, slice_fk_pose, normalize_pose, renormalize_pose, renormalize_joint_state
import pytorch_kinematics as pk
from .train import BaseTrainer
#from ranger import Ranger
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MLPTrainer(BaseTrainer):

    def __init__(self, args, trial=None, config=None, train_dataset=None, test_dataset=None):
        super().__init__(args, trial, config, train_dataset, test_dataset)

        if self.optimizer_run:
            self.position_weight = args.position_weight
            self.orientation_weight = args.orientation_weight
        else:
            self.position_weight = self.config["IKNet"]['position_weight']
            self.orientation_weight = self.config["IKNet"]['orientation_weight']

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
        self.zero_joints_loss = torch.nn.MSELoss(reduction='none').to(self.device)

        self.use_neg_quat = False

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

        self.gpu = args.gpu
        self.loss_history = [[],[],[]]

        if self.zero_joints_goal:
            self.zero_controller_weight = self.config["IKNet"]['zero_controller_weight']
            self.zero_controller_weight_tensor = torch.tensor(self.zero_controller_weight).to(self.device)
        self.suffix = None
        if args.suffix != '':
            self.suffix = args.suffix

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
                zero_joints_loss_ik = torch.mean(self.zero_joints_loss(joint_index_view, self.small_zero_joints_array), axis=0)
            else:
                zero_joints_loss_ik = torch.mean(self.zero_joints_loss(joint_index_view, self.zero_joints_array), axis=0)
                #print(zero_joints_loss_ik.size())
        else:
            zero_joints_loss_ik = torch.mean(self.zero_joints_loss(joint_index_view,
                                                        torch.zeros(len(joint_state), len(joint_index_view[0])).to(
                                                            self.device)))
        #zero_joints_loss_ik = torch.mean(torch.multiply(self.zero_controller_weight_tensor, zero_joints_loss_ik))
        return zero_joints_loss_ik

    def training_step(self, data, i):
        gt_A = data["gt_A"].to(self.device)
        gt_B = data["gt_B"].to(self.device)
        real_B = data["real_B"].to(self.device)
        bs = len(gt_B)

        #if i % 2 == 0:
        #    gt_B[:, 3:] = -gt_B[:, 3:]
        #    #real_B[:, 3:] = -real_B[:, 3:]

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
        #noise = torch.rand((len(gt_B), self.robot_dof), device=self.device) - 0.5
        #noise *= 0.05
        #backward_B2A =backward_B2A + noise

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

            #position_noise = torch.randn((len(gt_B), 3), device=self.device) - 0.5
            #orientation_noise = torch.randn((len(gt_B), 4), device=self.device) - 0.5
            #position_noise *= 0.0005
            #orientation_noise *= 0.002
            #forward_result[:, :3] = forward_result[:, :3] + position_noise
            #forward_result[:, 3:] = forward_result[:, 3:] + orientation_noise

            cycle_loss_B2A_position = self.position_cycle_loss(forward_result[:, :3], real_B[:, :3])
            cycle_loss_B2A_orientation = self.orientation_loss_phi_7(forward_result[:, 3:], real_B[:, 3:])

        zero_joints_loss_ik = None

        #self.loss_history[0].append(cycle_loss_B2A_position.clone().detach().cpu())
        #self.loss_history[1].append(cycle_loss_B2A_orientation.clone().detach().cpu())

        if self.zero_joints_goal:
            zero_joints_loss_ik = self.calculate_zero_joints_loss(js, bs)
            #if self.zero_joints_goal:
            #    self.loss_history[2].append(zero_joints_loss_ik.clone().detach().cpu())
            errG = cycle_loss_B2A_position * ( self.position_weight * 1000) + cycle_loss_B2A_orientation * (self.orientation_weight * 1000) + torch.mean(torch.multiply(self.zero_controller_weight_tensor, zero_joints_loss_ik))
            zero_joints_loss_ik = torch.mean(zero_joints_loss_ik)
        else:
            errG = cycle_loss_B2A_position * self.position_weight + cycle_loss_B2A_orientation * self.orientation_weight
            zero_joints_loss_ik = torch.Tensor([0])

        #errG.register_hook(lambda grad: print(grad))
        errG.backward()

        return cycle_loss_B2A_position, cycle_loss_B2A_orientation, zero_joints_loss_ik, cycle_loss_A2B_position, cycle_loss_A2B_orientation

    @torch.no_grad()
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
            cycle_loss_B2A_orientation = self.orientation_loss_phi_7(forward_result[:, 3:], real_B[:, 3:])

            return cycle_loss_B2A_position, cycle_loss_B2A_orientation, zero_joints_loss_ik, cycle_loss_A2B_position, cycle_loss_A2B_orientation

    def create_model(self):
        super().create_model()
        if self.optimizer_run:
            self.model = GenericGenerator(input_size=7, output_size=self.robot_dof,
                                                  nbr_tanh=self.nbr_tanh,
                                                  activation=self.activation,
                                                  layers=self.layers).to(self.device)
        else:
            self.model = GenericGenerator(input_size=7, output_size=self.robot_dof,
                                                  nbr_tanh=self.config["IKNet"]["architecture"]["nbr_tanh"],
                                                  activation=self.config["IKNet"]["architecture"]["activation"],
                                                  layers=self.config["IKNet"]["architecture"]["layers"]).to(self.device)

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

        if self.zero_joints_goal:
            loss_for_lr = position_error * 100 + orientation_error * 10 + variance_error
        else:
            loss_for_lr = position_error * 100 + orientation_error * 10
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
        #self.lr_scheduler = ReduceLROnPlateau(optimizer=self.optimizer, factor=0.6, mode='min')

    def create_optimizer(self):
        super().create_optimizer()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        #self.optimizer = Ranger(params=self.model.parameters(), lr=self.lr)

    def create_checkpoint(self, epoch):
        if epoch % 5 == 0 and not self.optimizer_run:
            if self.suffix is not None:
                torch.save(self.model.state_dict(), f"weights/{self.robot}/model_MLP_epoch_{epoch}_with_kinematics_{self.suffix }.pth")
            else:
                torch.save(self.model.state_dict(), f"weights/{self.robot}/model_MLP_epoch_{epoch}_with_kinematics_gpu{self.gpu}.pth")

    def train(self):
        torch.cuda.empty_cache()

        if self.compile:

            print(f'\n\n {torch._dynamo.list_backends()} \n\n')

            training_step_compiled = torch.compile(self.training_step, backend='inductor', dynamic=False)
            process_training_losses_compiled = self.process_training_losses#torch.compile(self.process_training_losses, backend='inductor')
            validation_step_compiled = torch.compile(self.validation_step, backend='inductor', dynamic=False)
            process_validation_losses_compiled = self.process_validation_losses#torch.compile(self.process_validation_losses, backend='inductor')

            print(f'\n\n {torch._dynamo.utils.compile_times()} \n\n')

        else:
            training_step_compiled = self.training_step
            process_training_losses_compiled = self.process_training_losses
            validation_step_compiled = self.validation_step
            process_validation_losses_compiled = self.process_validation_losses

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
                    cycle_loss_A2B_position, cycle_loss_A2B_orientation = training_step_compiled(data, i)
                self.optimizer.step()
                if self.autoencoder and not self.two_stage:
                    self.fk_optimizer.step()
                process_training_losses_compiled(cycle_loss_B2A_position, cycle_loss_B2A_orientation,
                                             variance_loss_B2A, cycle_loss_A2B_position,
                                             cycle_loss_A2B_orientation, len(data["gt_B"]), epoch)

            self.position_losses.append(self.position_loss / len(self.train_dataloader))
            self.orientation_losses.append(self.orientation_loss / len(self.train_dataloader))
            self.fk_position_losses.append(self.fk_position_loss / len(self.train_dataloader))
            self.fk_orientation_losses.append(self.fk_orientation_loss / len(self.train_dataloader))
            self.var_losses.append(self.variance_loss / len(self.train_dataloader))

            #self.create_checkpoint(epoch)

            self.val_position_loss = 0
            self.val_orientation_loss = 0
            self.val_variance_loss = 0
            self.val_fk_position_loss = 0
            self.val_fk_orientation_loss = 0

            if epoch == 0 or (epoch + 1) % 1 == 0:
                self.progress_bar = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader))
                for i, data in self.progress_bar:
                    # get batch size data
                    cycle_loss_B2A_val_position, cycle_loss_B2A_val_orientation, variance_loss_B2A, \
                        cycle_loss_A2B_position, cycle_loss_A2B_orientation = validation_step_compiled(data)
                    process_validation_losses_compiled(cycle_loss_B2A_val_position,
                                                   cycle_loss_B2A_val_orientation, variance_loss_B2A,
                                                   cycle_loss_A2B_position, cycle_loss_A2B_orientation,
                                                   len(data["gt_B"]), epoch)

            self.val_varience_losses.append(self.val_variance_loss / len(self.test_dataloader))
            self.val_orientation_losses.append(self.val_orientation_loss / len(self.test_dataloader))
            self.val_position_losses.append(self.val_position_loss / len(self.test_dataloader))
            self.val_fk_position_losses.append(self.val_fk_position_loss / len(self.test_dataloader))
            self.val_fk_orientation_losses.append(self.val_fk_orientation_loss / len(self.test_dataloader))

            avg_loss = self.average_loss / len(self.train_dataloader)

            print("Avg Loss: {0}".format(avg_loss))

            if self.optimizer_run:
                self.trial.report(avg_loss, epoch)

                if self.trial.should_prune():
                    raise optuna.TrialPruned()

            #self.lr_scheduler.step(((self.position_losses[-1] * 10) + self.orientation_losses[-1]))
            #print((epoch % 2 == 1))
            self.lr_scheduler.step()
                #print(self.lr_scheduler.get_lr())

        print([self.position_losses, self.orientation_losses, self.var_losses,
               self.val_position_losses, self.val_orientation_losses, self.val_varience_losses])

        if not self.optimizer_run:
            # save last check pointing
            if self.suffix is not None:
                torch.save(self.model.state_dict(), f"weights/{self.robot}/model_MLP_with_kinematics_{self.suffix}.pth")
                with open(rf"results/{self.robot}/train_MLP_with_kinematics_loss_{self.suffix}.p", "wb") as output_file:
                    pickle.dump([self.position_losses, self.orientation_losses, self.var_losses,
                                 self.val_position_losses, self.val_orientation_losses, self.val_varience_losses], output_file)
            else:
                torch.save(self.model.state_dict(), f"weights/{self.robot}/model_MLP_with_kinematics_gpu{self.gpu}.pth")
                with open(rf"results/{self.robot}/train_MLP_with_kinematics_loss_gpu{self.gpu}.p", "wb") as output_file:
                    pickle.dump([self.position_losses, self.orientation_losses, self.var_losses,
                   self.val_position_losses, self.val_orientation_losses, self.val_varience_losses], output_file)

        return (self.val_position_losses[-1] * 100) + (self.val_orientation_losses[-1] * 10) + (self.val_varience_losses[-1])