import argparse
from test import MLPTester, FKTester, MoveitTester


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
    parser.add_argument("--cuda", action="store_true", help="Enables cuda")
    parser.add_argument("--test_moveit", action="store_true", help="Enables cuda")
    parser.add_argument("--use_ga", action="store_true", help="Enables cuda")
    parser.add_argument("--use_optimizer", action="store_true", help="Enables cuda")
    parser.add_argument("--manualSeed", type=int,
                        help="Seed for initializing training. (default:none)")
    parser.add_argument("--robot", type=str, default="nicol", help="Robot model IK is trained for")
    parser.add_argument("--gpu", type=int, default=0, help="Robot model IK is trained for")
    parser.add_argument("--network", type=str, default="", help="Robot model IK is trained for")
    parser.add_argument("--core-model", type=str, default="nicol", help="Robot model IK is trained for")
    parser.add_argument("--autoencoder", action="store_true", help="Enables learned FK")
    parser.add_argument("--two-stage", action="store_true", help="Enables two-stage learned FK training")
    parser.add_argument("--finetune", action="store_true", help="Enables two-stage learned FK training")
    parser.add_argument("--chain", type=str, default="right_arm", help="Robot model Kinematic Chain")
    parser.add_argument("--ik_name", type=str, default="bioik", help="Robot model Kinematic Chain")

    #print(args)

    test_args = parser.parse_args()

    tester = None

    if test_args.network == "MLP":
        tester = MLPTester(test_args)
    elif test_args.network == "FK":
        tester = FKTester(test_args)
    elif test_args.test_moveit:
        tester = MoveitTester(test_args)

    tester.test()
