# Copyright (c) 2023. Jan-Gerrit Habekost. GNU General Public License https://www.gnu.org/licenses/gpl-3.0.html.

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
args = parser.parse_args()

loaded_study = optuna.load_study(study_name="cycleik_ik_optimizer", storage=f'sqlite:///{args.db}')
joblib.dump(loaded_study, f"./optuna/{args.robot}/cycleik_ik_optimizer.pkl")
print(len(loaded_study.trials))
sorted_trials = sorted(loaded_study.trials, key=get_score)

for i in range(10):
    print(sorted_trials[i])

print("Best Config:\n {0}".format(loaded_study.trials[4]))