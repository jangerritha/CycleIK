from cycleik_pytorch import CycleIK
import pickle
import argparse

parser = argparse.ArgumentParser(
    description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")

parser.add_argument("--robot", type=str, default="nicol", help="Robot model IK is trained for")
parser.add_argument("--gpu", type=str, default="0", help="GPU used for training")
parser.add_argument("--samples", type=int, default="1000000", help="GPU used for training")
args = parser.parse_args()

def main():

    robot_name = args.robot
    data_file_name = f"results_{robot_name}_{int(args.samples / 1000)}"

    cycle_ik = CycleIK(robot=robot_name, cuda_device=args.gpu)
    random_points = cycle_ik.get_random_samples(args.samples)
    #print(random_points[1][5000])
    with open(f'/home/jan-gerrit/repositories/cycleik/data/{data_file_name}.p', 'wb') as f:
        pickle.dump(random_points, f)
        f.close()

if __name__ == '__main__':
    main()