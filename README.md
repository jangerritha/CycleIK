# CycleIK

## Overview
<p align="center">
Implementation of CycleIK - Neuro-inspired Inverse Kinematics

<img src="/assets/img/_DSC0957__.JPG"  height="320"><br>Example output of the CycleIK GAN for the Neuro-Inspired Collaborator (NICOL)
</p>

<p align="center">
<img src="/assets/img/cycle_ik_overview.jpg"  height="320"><br>Neuro-inspired inverse kinematics training architecture. A batch of
poses X is inferenced by either the MLP or GAN to obtain the batch of joint angles
Θ. The joint angles are transformed back to Cartesian space X by the FK(Θ)
function to be evaluated by the multi-objective function under constraints L.
</p>

### Publications
```
Inverse Kinematics for Neuro-Robotic Grasping with Humanoid Embodied Agents. (2024). 
Habekost, JG., Strahl, E., Allgeuer, P., Kerzel, M., Wermter, S. 
Accepted at IROS 2024.
```
Arxiv: https://arxiv.org/html/2404.08825

```
CycleIK: Neuro-inspired Inverse Kinematics. (2023). 
Habekost, JG., Strahl, E., Allgeuer, P., Kerzel, M., Wermter, S. 
In: Artificial Neural Networks and Machine Learning – ICANN 2023. 
Lecture Notes in Computer Science, vol 14254. 
```
Arxiv: https://arxiv.org/abs/2307.11554 \
Open Access: https://link.springer.com/chapter/10.1007/978-3-031-44207-0_38


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


## Model Description
Two neuro-inspired IK models are available. An MLP that deterministically returns a single solution to a single IK query. 
In addition a GAN is available that offers exploration of the nullspace to produce multiple solutions for a single IK query.

### MLP
<img src="/assets/img/generator.png"  height="320"><br>*MLP IK model optimized for NICOL robot*

### GAN
<img src="/assets/img/generator_gan.png"  height="320"><br>*GAN IK model optimized for NICOL robot*

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 

### Credit


