# CycleIK

## Overview
Implementation of CycleIK 

Paper: https://arxiv.org/abs/2307.11554

Accepted at ICANN 23 (32nd International Conference on Artificial Neural Networks)

<img src="/assets/img/example_GAN_nicol.png"  height="320"><br>*Example output of the CycleIK GAN for the Neuro-Inspired Collaborator (NICOL)*

## Hybrid Neuro-Genetic IK Pipeline
<img src="/assets/img/cycleik_ik_pipeline.jpg"  height="320"><br>*Hybrid neuro-genetic IK pipeline. The neural solutions for pose X can
optionally be optimized with the Gaikpy GA and SLSQP, given the constraints L.*

### Installation

#### Clone and install requirements

```bash
git clone -b dev https://git.informatik.uni-hamburg.de/jangerritha/cycleik.git
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

#### Download pretrained weights

```bash

```

#### Download dataset

```bash

```

### Test

The following commands can be used to test the whole test.

```bash

```

For single image processing, use the following command.

```bash

```


### Train

```text
usage: train.py [-h] [--dataroot DATAROOT] [--dataset DATASET] [--epochs N]
                [--decay_epochs DECAY_EPOCHS] [-b N] [--lr LR] [-p N] [--cuda]
                [--netG_A2B NETG_A2B] [--netG_B2A NETG_B2A] [--netD_A NETD_A]
                [--netD_B NETD_B] [--image-size IMAGE_SIZE] [--outf OUTF]
                [--manualSeed MANUALSEED]

PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent
Adversarial Networks`

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   path to datasets. (default:./data)
  --dataset DATASET     dataset name. (default:`horse2zebra`)Option:
                        [apple2orange, summer2winter_yosemite, horse2zebra,
                        monet2photo, cezanne2photo, ukiyoe2photo,
                        vangogh2photo, maps, facades, selfie2anime,
                        iphone2dslr_flower, ae_photos, ]
  --epochs N            number of total epochs to run
  --decay_epochs DECAY_EPOCHS
                        epoch to start linearly decaying the learning rate to
                        0. (default:100)
  -b N, --batch-size N  mini-batch size (default: 1), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel
  --lr LR               learning rate. (default:0.0002)
  -p N, --print-freq N  print frequency. (default:100)
  --cuda                Enables cuda
  --netG_A2B NETG_A2B   path to netG_A2B (to continue training)
  --netG_B2A NETG_B2A   path to netG_B2A (to continue training)
  --netD_A NETD_A       path to netD_A (to continue training)
  --netD_B NETD_B       path to netD_B (to continue training)
  --image-size IMAGE_SIZE
                        size of the data crop (squared assumed). (default:256)
  --outf OUTF           folder to output images. (default:`./outputs`).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:none)

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

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 

### Credit


