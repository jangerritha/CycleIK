import argparse
from optimize import FKOptimizer, MLPOptimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
    parser.add_argument("--robot", type=str, default="nicol", help="Robot model IK is trained for")
    parser.add_argument("--chain", type=str, default="right_arm", help="Robot model Kinematic Chain")
    parser.add_argument("--network", type=str, default="MLP", help="Network architecture")
    parser.add_argument("--study_name", type=str, default="", help="Network architecture")
    parser.add_argument("--trials", default=100, type=int, metavar="N", help="number of total trials in optimizer")
    parser.add_argument("--epochs", default=10, type=int, metavar="N", help="number of total trials in optimizer")
    parser.add_argument("--gpu", type=str, default="0", help="GPU used for training")
    parser.add_argument("--autoencoder", action="store_true", help="Enables autoencoder")
    parser.add_argument("--two-stage", action="store_true", help="Enables two-stage training for autoencoder")
    parser.add_argument("--db", type=str, default="ik_optimizer_results.db", help="GPU used for training")
    parser.add_argument("--add_seed", action="store_true", help="GPU used for training")
    parser.add_argument("--compile", action="store_true", help="Enables pytorch compiling")
    opt_args = parser.parse_args()

    optimizer = None

    if opt_args.network == "MLP":
        optimizer = MLPOptimizer(opt_args)
    elif opt_args.network == "FK":
        optimizer = FKOptimizer(opt_args)

    optimizer.optimize()
