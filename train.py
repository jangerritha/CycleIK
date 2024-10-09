import argparse
from typing import List

from train import MLPTrainer, FKTrainer, FineTuneMLPTrainer, MLPMultiRun


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
    parser.add_argument("--dataroot", type=str, default="./data",
                        help="path to datasets. (default:./data)")
    parser.add_argument("--dataset", type=str, default="horse2zebra",
                        help="dataset name. (default:`horse2zebra`)"
                             "Option: [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, "
                             "cezanne2photo, ukiyoe2photo, vangogh2photo, maps, facades, selfie2anime, "
                             "iphone2dslr_flower, ae_photos, ]")
    parser.add_argument("--epochs", default=100, type=int, metavar="N",
                        help="number of total epochs to run")
    parser.add_argument("--decay_epochs", type=int, default=0,
                        help="epoch to start linearly decaying the learning rate to 0. (default:100)")
    parser.add_argument("--runs", type=int, default=1,
                        help="epoch to start linearly decaying the learning rate to 0. (default:100)")
    parser.add_argument("-b", "--batch-size", default=None, type=int,
                        metavar="N",
                        help="mini-batch size (default: 1), this is the total "
                             "batch size of all GPUs on the current node when "
                             "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--lr", type=float, default=None,
                        help="learning rate. (default:0.0002)")
    parser.add_argument("--fk_lr", type=float, default=None,
                        help="learning rate. (default:0.0002)")
    parser.add_argument("--noise_vector_size", type=int, default=None,
                        help="learning rate. (default:0.0002)")
    parser.add_argument("--noise", type=float, default=None,
                        help="noise on position for fk. (default:0.0002)")
    parser.add_argument("-p", "--print-freq", default=100, type=int,
                        metavar="N", help="print frequency. (default:100)")
    parser.add_argument("--cuda", action="store_true", help="Enables cuda")
    parser.add_argument("--her", action="store_true", help="Enables Hindsight Experience Replay (HER)")
    parser.add_argument("--netG_A2B", default="", help="path to netG_A2B (to continue training)")
    parser.add_argument("--netG_B2A", default="", help="path to netG_B2A (to continue training)")
    parser.add_argument("--netD_A", default="", help="path to netD_A (to continue training)")
    parser.add_argument("--netD_B", default="", help="path to netD_B (to continue training)")

    parser.add_argument("--enc_A", default="", help="path to netD_A (to continue training)")
    parser.add_argument("--enc_B", default="", help="path to netD_B (to continue training)")

    parser.add_argument("--image-size", type=int, default=256,
                        help="size of the data crop (squared assumed). (default:256)")
    parser.add_argument("--outf", default="./outputs",
                        help="folder to output images. (default:`./outputs`).")
    parser.add_argument("--manualSeed", type=int,
                        help="Seed for initializing training. (default:none)")
    parser.add_argument("--robot", type=str, default="nicol", help="Robot model IK is trained for")
    parser.add_argument("--gpu", type=str, default="0", help="GPU used for training")
    parser.add_argument("--gpus", nargs='+', type=int, default=[0], help="GPU used for training")
    parser.add_argument("--network", type=str, default="MLP", help="Network type to be trained")
    parser.add_argument("--autoencoder", action="store_true", help="Enables autoencoder")
    parser.add_argument("--two-stage", action="store_true", help="Enables two-stage training for autoencoder")
    parser.add_argument("--finetune", action="store_true", help="Enables two-stage training for autoencoder")
    parser.add_argument("--multi-run", action="store_true", help="Enables two-stage training for autoencoder")
    parser.add_argument("--core-model", type=str, default="nicol", help="Enables two-stage training for autoencoder")
    parser.add_argument("--chain", type=str, default="right_arm", help="Robot model Kinematic Chain")
    parser.add_argument("--core_model_chain", type=str, default="right_arm", help="Robot model Kinematic Chain")
    parser.add_argument("--suffix", type=str, default="", help="suffix for persistent weights and stats")
    parser.add_argument("--compile", action="store_true", help="Enables pytorch compiling")
    train_args = parser.parse_args()



    if train_args.multi_run:
        multi_runner = None
        if train_args.network == "GAN":
            raise NotImplementedError
        elif train_args.network == "MLP":
            multi_runner = MLPMultiRun(train_args)
        multi_runner.run()
    else:
        trainer = None
        if train_args.network == "MLP":
            if train_args.finetune:
                trainer = FineTuneMLPTrainer(train_args)
            else:
                trainer = MLPTrainer(train_args)
        elif train_args.network == "FK":
            trainer = FKTrainer(train_args)
        trainer.train()
