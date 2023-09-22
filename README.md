# CycleIK

## Overview
Implementation of CycleIK, ICANN 23 version to reproduce the results

For the updated architecture that offers full PyTorch support for the whole Neuro-Genetic IK Pipeline, please see branch: [`dev`](https://git.informatik.uni-hamburg.de/jangerritha/cycleik.git)

Paper: https://arxiv.org/abs/2307.11554

Accepted at ICANN 23 (32nd International Conference on Artificial Neural Networks)

<br>
<br>

<img src="/assets/img/example_GAN_nicol.png"  height="310"><br>*Example output of the CycleIK GAN for the Neuro-Inspired Collaborator (NICOL)*

<br>
<br>

<img src="/assets/img/overlay_real_nicol.png"  height="320"><br>*CycleIK MLP deployed to physical Neuro-Inspired Collaborator (NICOL)*

<br>
<br>

------------------------------------------------------------------------------------------------------------------------

## Installation

```bash
git clone -b ICANN23_version https://git.informatik.uni-hamburg.de/jangerritha/cycleik.git
cd cycleik/
pip install -r requirements.txt
cd ..
git clone https://github.com/UM-ARM-Lab/pytorch_kinematics.git
cd pytorch_kinematics
git checkout bee97ab0d20d21145df2519b1a6db346f20f78d4
pip install -e .
cd ../cycleik
pip install -e .
```

------------------------------------------------------------------------------------------------------------------------

## Experiment Setup

### Download dataset
Download the following datasets and place them in `<path_to_repo>/data/`

#### Small Workspace
[Training Data](https://drive.google.com/file/d/1aniD5yotkjCtwx-A6wW7SXaSxZPvQYAt/view?usp=sharing)

[Test Data](https://drive.google.com/file/d/1nyhTPRSfbSoKTG9FL2KHO0xo-_VSM_H7/view?usp=sharing)

[Validation Data](https://drive.google.com/file/d/16zZaEhE4gTDME8pEaeqMn52g-sk9elis/view?usp=sharing)


#### Full Workspace
[Training Data](https://drive.google.com/file/d/19jP3gQxUCFXWFN_5VJv-NfG8gfZvPOjH/view?usp=sharing)

[Test Data](https://drive.google.com/file/d/1fU5Dad5E2UP-klXnmpcJMjn_FyKNk4jU/view?usp=sharing)

[Validation Data](https://drive.google.com/file/d/15A4nCzJcyvWZpKnr2bAPTvhyQJVDxHD3/view?usp=sharing)

------------------------------------------------------------------------------------------------------------------------

### Download pretrained weights (best models)

Download the following pre-trained model weights and place them in `<path_to_repo>/weights/nicol/`

#### MLP 
[Small workspace](https://drive.google.com/file/d/19DsbI91V2r8QvHONJflKbGAyO2PEKD8P/view?usp=sharing)

[Full workspace](https://drive.google.com/file/d/1Z57bRYDjpos2cEPTnZZQplaKBsAANS1_/view?usp=sharing)

#### GAN
[Small workspace](https://drive.google.com/file/d/16wTyCIWuJutKUOUsQqyz5nqAVX4Cdrvb/view?usp=sharing)

[Full workspace](https://drive.google.com/file/d/1VA4D9hMMjZRcrDqBLz_pC3GCronS_djb/view?usp=sharing)

------------------------------------------------------------------------------------------------------------------------

### Config

Configure the config file under `<path_to_repo>/config/nicol.yaml` for either the `Small` or the `Full` workspace

#### Small Workspace

```
train_data: 'results_nicol_1000_3'
test_data: 'results_nicol_10_3'
val_data: 'results_nicol_100_3'

robot_dof: 8
limits:
  upper: [2.5, 1.8, 1.5, 2.9, 1.570796, 3.141592, 0.785398, 0.785398]
  lower: [0., -1.5, -2.25, -2.9, -1.570796, -3.141592, -0.785398, -0.785398]

workspace:
  upper: [0.85, 0.0, 1.4]
  lower: [0.2, -0.9, 0.8]


robot_urdf: './assets/urdf/NICOL.urdf'
robot_eef: 'r_laser'

zero_joints_goal: [2, 3, 4, 5]

FKNet:
  training:
    batch_size: 700
    lr: 0.0001

  architecture:
    layers:  [1900, 2700, 3000, 2900, 450, 60, 10, 160]
    nbr_tanh: 3
    activation: "GELU"

IKNet:
  training:
    batch_size: 150
    lr: 0.00016

  architecture:
    layers:  [3380, 2250, 3240, 2270, 1840, 30, 60, 220]
    nbr_tanh: 3
    activation: "GELU"

GAN:
  training:
    batch_size: 350
    lr: 0.00021

  architecture:
    noise_vector_size: 8
    layers: [ 790, 990, 3120, 1630, 300, 1660, 730, 540 ]
    nbr_tanh: 3
    activation: "GELU"
```

#### Full Workspace

```
train_data: 'results_nicol_1400_5'
test_data: 'results_nicol_14_5'
val_data: 'results_nicol_140_5'

robot_dof: 8
limits:
  upper: [2.5, 1.8, 1.5, 2.9, 1.570796, 3.141592, 0.785398, 0.785398]
  lower: [0., -1.5, -2.25, -2.9, -1.570796, -3.141592, -0.785398, -0.785398]

workspace:
  upper: [0.85, 0.48, 1.4]
  lower: [0.2, -0.9, 0.8]


robot_urdf: './assets/urdf/NICOL.urdf'
robot_eef: 'r_laser'

zero_joints_goal: [3, 4]

FKNet:
  training:
    batch_size: 700
    lr: 0.0001

  architecture:
    layers:  [1900, 2700, 3000, 2900, 450, 60, 10, 160]
    nbr_tanh: 3
    activation: "GELU"

IKNet:
  training:
    batch_size: 300
    lr: 0.0001

  architecture:
    layers:  [2200, 2400, 2400, 1900, 250, 220, 30, 380]
    nbr_tanh: 3
    activation: "GELU"

GAN:
  training:
    batch_size: 300
    lr: 0.00019

  architecture:
    noise_vector_size: 10
    layers: [ 1180, 1170, 2500, 1290, 700, 970, 440, 770 ]
    nbr_tanh: 2
    activation: "GELU"
```

------------------------------------------------------------------------------------------------------------------------

## Test

It is recommended to run the tests on the GPU by using the `--cuda` flag

### MLP

```
    python test_ik.py --cuda
```

### GAN

```
    python test_GAN.py --cuda
```


## Train

It is recommended to run the training on the GPU by using the `--cuda` flag

### MLP

```
    python train_ik.py --cuda
```

### GAN

```
    python train_GAN.py --cuda
```

## Model Description
Two neuro-inspired IK models are available. An MLP that deterministically returns a single solution to a single IK query. 
In addition a GAN is available that offers exploration of the nullspace to produce multiple solutions for a single IK query.

### MLP
<img src="/assets/img/generator.png"  height="320"><br>*MLP IK model optimized for NICOL robot*

### GAN
<img src="/assets/img/generator_gan.png"  height="320"><br>*GAN IK model optimized for NICOL robot*

## Training Algorithm
<img src="/assets/img/cycle_ik_overview.jpg"  height="320"><br>*Neuro-inspired inverse kinematics training architecture. A batch of
poses X is inferenced by either the MLP or GAN to obtain the batch of joint angles
Θ. The joint angles are transformed back to Cartesian space X by the FK(Θ)
function to be evaluated by the multi-objective function under constraints L.*
