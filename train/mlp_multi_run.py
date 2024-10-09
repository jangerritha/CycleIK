#  Copyright (c) 2024. Jan-Gerrit Habekost. GNU General Public License. https://www.gnu.org/licenses/gpl-3.0.html.
import torch.utils.data
from cycleik_pytorch import DecayLR, IKDataset, GenericNoisyGenerator
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
import copy
import torch.multiprocessing as mp
from train import MLPTrainer
import time
#torch.multiprocessing.set_start_method('forkserver', force=True)

def train_model(gpu_queue, run_queue, args):
    gpu = gpu_queue.get(block=True)
    print(f'gpu: {gpu}')

    keep_running = run_queue.qsize() > 0

    while keep_running:
        next_suffix = run_queue.get(block=False)
        temp_args = copy.deepcopy(args)
        temp_args.suffix = copy.deepcopy(next_suffix)
        temp_args.gpu = copy.deepcopy(gpu)
        print(f'next suffix: {next_suffix}')
        del next_suffix

        temp_trainer = MLPTrainer(temp_args)
        temp_trainer.train()

        keep_running = run_queue.qsize() > 0


    del gpu
        # del next_suffix

class MLPMultiRun():

    def __init__(self, args):
        self.args = args
        self.runs = args.runs
        self.gpus = self.args.gpus
        print(print(f'gpus: {self.gpus}'))
        self.suffixes = []
        for i in range(self.runs):
            self.suffixes.append(f'IROS_train_run_{i}')



    def run(self):
        ctx = mp.get_context('spawn')
        gpu_queue = ctx.Queue()
        run_queue = ctx.Queue()
        for gpu in self.gpus:
            gpu_queue.put(gpu)
        for suffix in self.suffixes:
            run_queue.put(suffix)

        print(gpu_queue)

        processes = []

        for i in range(len(self.gpus)):
            process = ctx.Process(target=train_model, args=(gpu_queue, run_queue, self.args))
            process.start()
            processes.append(process)
        for p in processes:
            p.join()