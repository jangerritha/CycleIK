import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import joblib

import optuna

# You can use Matplotlib instead of Plotly for visualization by simply replacing `optuna.visualization` with
# `optuna.visualization.matplotlib` in the following examples.
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


def main():
    study = joblib.load('./optuna/cycleik_fk_optimizer.pkl')
    plot_optimization_history(study).show()
    #plot_parallel_coordinate(study).show()
    plot_contour(study, params=["lr", "activation", "batch_size"]).show()
    plot_contour(study, params=["lr", "activation", "nbr_layers"]).show()
    plot_contour(study, params=["lr", "batch_size", "nbr_layers"]).show()
    plot_slice(study, params=["layer0_neurons", "layer1_neurons", "layer2_neurons", "layer3_neurons", "layer6_neurons", "layer7_neurons", "layer8_neurons", "layer9_neurons"]).show()
    plot_slice(study,
               params=["lr", "activation", "batch_size", "nbr_layers", "nbr_tanh"]).show()
    plot_param_importances(study).show()
    print(study.best_trial)

if __name__ == '__main__':
    main()