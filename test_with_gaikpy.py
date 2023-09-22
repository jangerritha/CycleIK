# Copyright (c) 2023. Jan-Gerrit Habekost. GNU General Public License https://www.gnu.org/licenses/gpl-3.0.html.

import argparse
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from tqdm import tqdm
import pickle
np.set_printoptions(linewidth=np.inf)
from cycleik_pytorch import Generator, IKDataset, Discriminator
from cycleik_pytorch import get_kinematic_params, load_config


parser = argparse.ArgumentParser(
    description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Dataset

batch_size=66354

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
fk_model = Discriminator(input_size=8, output_size=7).to(device)
model = Generator(input_size=7, output_size=8).to(device)

# Load state dicts
fk_model.load_state_dict(torch.load(os.path.join("weights", "nicol", "netG_A2B_2000.pth")))
model.load_state_dict(torch.load(os.path.join("weights", "nicol", "netG_B2A_2000.pth")))

# Set model mode
fk_model.eval()
model.eval()

fk_error = [0, 0, 0]
ik_error = [0, 0, 0]


min_error = 10000
max_error = 0
avg_error = 0

min_error_ik = 10000
max_error_ik = 0
avg_ik_error = 0

over_1mm = 0
over_2_5mm = 0
over_5mm = 0
over_1cm = 0
over_1_5cm = 0
over_2cm = 0
over_5cm = 0
over_10cm = 0

over_1mm_ik = 0
over_2_5mm_ik = 0
over_5mm_ik = 0
over_1cm_ik = 0
over_1_5cm_ik = 0
over_2cm_ik = 0
over_5cm_ik = 0
over_10cm_ik = 0

it_counter = 0

config = load_config("nicol")
workspace, limits_upper, limits_lower, normalize_interval_array, normalize_center_array = get_kinematic_params(config)

data_path = str(os.getcwd())
with open(data_path + "/data/validation_dataset_gaikpy_nicol.p", 'rb') as f:
    loaded_values = pickle.load(f)

progress_bar = tqdm(enumerate(loaded_values), total=len(loaded_values))

output_data = []
test_error = 0
bigger_2cm_test_error = 0

nbr_bigger_2cm = 0

for i, (pose, js) in progress_bar:
    normalized_pose = np.copy(pose)
    normalized_pose[2] = normalized_pose[2] / 1.8
    normalized_js = np.copy(js)
    for k in range(6):
        normalized_js[k] = normalized_js[k] / np.pi
    with torch.no_grad():
        result_pose = fk_model(torch.Tensor(normalized_js).to(device)).cpu().detach().numpy()
        result_js = model(torch.Tensor(normalized_pose).to(device)).cpu().detach().numpy()
    result_pose[2] = result_pose[2] * 1.8

    temp_error = np.abs(np.subtract(result_pose, pose))
    if np.average(temp_error) > 0.025:
        if js[0] > 0.:
            print(js)
        bigger_2cm_test_error += temp_error
        nbr_bigger_2cm += 1
    else:
        test_error += temp_error


    for l in range(6):
        result_js[l] = result_js[l] * np.pi

    output_data.append([pose, js, result_pose, result_js])

test_error = test_error / (10000 - nbr_bigger_2cm)
bigger_2cm_test_error = bigger_2cm_test_error / nbr_bigger_2cm

print("\nNumber of samples with > 2.5cm error: {0} \n".format(nbr_bigger_2cm))
print("Ratio of samples with > 2.5cm error: {0} \n".format(nbr_bigger_2cm / 10000))
print("Avg. test error for examples with =< 2.5cm position error:  \n {0} \n".format(test_error))
print("Avg. test error for examples with > 2.5cm position error: \n {0} \n".format(bigger_2cm_test_error))

with open(data_path + "/data/validation_dataset_gaikpy_nicol_results.p", 'wb') as f:
    pickle.dump(output_data, f)
