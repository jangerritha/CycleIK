# CycleIK

## Overview
<p align="center">
Implementation of CycleIK - Neuro-inspired Inverse Kinematics
</p>

<p align="center">
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

Best practice is to use anaconda or miniconda

```bash
git clone -b dev https://git.informatik.uni-hamburg.de/jangerritha/cycleik.git
cd cycleik/
pip install -r requirements.txt
pip install -e .
```
### Usage

```bash
cycleik = CycleIK(robot="nicol", cuda_device='0', verbose=True, chain="right_arm")

start_pose = np.array([0.3, -0.5, 0.99, 0.0015305, 0.0009334, 0.70713, 0.70708])

target_pose = np.array([0.6, -0.3, 0.895, -0.5, 0.5, 0.5, 0.5])

control_points = np.array([np.array([0.3, -0.5, 1.2, -0.5, 0.5, 0.5, 0.5]),
                           np.array([0.6, -0.3, 1.2, -0.5, 0.5, 0.5, 0.5])])

bezier_curve = cycleik.generate_cubic_bezier_trajectory(start_pose=start_pose,
                                                              target_pose=target_pose,
                                                              control_points=control_points,
                                                              points=100)

joint_trajectory, _, _, _ = cycleik.inverse_kinematics(bezier_curve, calculate_error=True)
```


### Test
The MLP network can be tested with the following command, remove the `--cuda` flag incase you are running a CPU-only system.

Exchange `<robot>` by one of the following options: `nicol, nico, fetch, panda, or valkyrie`.

The `<chain>` tag refers to the kinematic chain that is selected for evaluation. \
Must match one of the move groups in the config yaml of the corresponding robot.

```bash
python test.py --cuda --robot <robot> --chain <chain> --network MLP 
```

#### Example
```bash
python test.py --cuda --robot nicol --chain right_arm --network MLP 
```
#### Download pretrained weights

Pre-trained weights are available for each of the five robot platforms on which CycleIK was evaluated.


```bash
cd <path_to_repo_location>
mkdir weights/<robot>
cd weights
mkdir <robot>
```

You can download a zip that contains all pre-trained weights from [here](https://drive.google.com/file/d/1SdZVdi4KtpBleBPvAcVP9s_GOpJ6zcOt/view?usp=sharing).

### Train
The training is executed very similar to running the tests.

```bash
python train.py --cuda --robot <robot> --chain <chain> --network MLP --epochs 10
```

#### Example
```bash
python train.py --cuda --robot nicol --chain right_arm --network MLP --epochs 10
```


#### Download dataset

For every robot there is a `train, test, and validation` dataset. Every dataset is delivered in a single file and must be added under `<path_to_repo_location>/data`.

```bash
cd <path_to_repo_location>
mkdir data
```

You can download a zip that contains all datasets used for our publication from [here](https://drive.google.com/file/d/1wc-YI9v0aEh0V0k5YqABckaJdNRqnNy7/view?usp=sharing).





### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.