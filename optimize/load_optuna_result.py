import argparse
import optuna
import joblib

def get_score(trial):
    if trial.values is not None:
        val =  trial.values[0]
        return val
    else:
        return 10000

parser = argparse.ArgumentParser(
    description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
parser.add_argument("--robot", type=str, default="nicol", help="Robot model IK is trained for")
parser.add_argument("--db", type=str, default="ik_optimizer_results.db", help="GPU used for training")
parser.add_argument("--gpu", type=int, default="0", help="GPU used for training")
parser.add_argument("--study_name", type=str, default="", help="Network architecture")
args = parser.parse_args()

if args.study_name != "":
    study_name = args.study_name
else:
    study_name = f"cycleik_ik_optimizer_{args.robot}_robot_gpu_{args.gpu}"


study_summaries = optuna.study.get_all_study_summaries(storage=f'sqlite:///{args.db}')

print(f"Optuna study summary: \n")

for i, study in enumerate(study_summaries):
    print(f'Study {i}: {study.study_name}\n')


loaded_study = optuna.load_study(study_name=study_name,
                                 storage=f'sqlite:///{args.db}')
joblib.dump(loaded_study, f"./optuna/{args.robot}/cycleik_ik_optimizer.pkl")
print(len(loaded_study.trials))
sorted_trials = sorted(loaded_study.trials, key=get_score)

for i in range(10):
    print(sorted_trials[i])

print("Best Config:\n {0}".format(loaded_study.trials[4]))