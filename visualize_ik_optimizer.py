# Copyright (c) 2023. Jan-Gerrit Habekost. GNU General Public License https://www.gnu.org/licenses/gpl-3.0.html.

import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import joblib

import optuna

# You can use Matplotlib instead of Plotly for visualization by simply replacing `optuna.visualization` with
# `optuna.visualization.matplotlib` in the following examples.
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice

import matplotlib.pyplot as plt
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 10}
plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=10)

def main():
    study = joblib.load('./optuna/nicol/OPTIMIZER_SAVES/1000_dataset/cycleik_ik_optimizer.pkl')
    #study = optuna.load_study(study_name="cycleik_ik_optimizer", storage=f'sqlite:///optuna/nicol/OPTIMIZER_SAVES/1400_dataset/ik_optimizer_results.db')

    print(len(study.trials))
    ### optimization history ###
    opt_history_subplot = plot_optimization_history(study, target_name='Mean Absolute Error')
    opt_history_subplot.set_ylim(top=0.01, bottom=0.0)
    opt_history_subplot.xaxis.label.set_fontsize(18)
    opt_history_subplot.xaxis.label.set_fontweight('bold')
    opt_history_subplot.yaxis.label.set_fontsize(18)
    opt_history_subplot.yaxis.label.set_fontweight('bold')
    opt_history_subplot.tick_params(axis='x', labelsize=14)
    opt_history_subplot.tick_params(axis='y', labelsize=14)
    opt_history_subplot.set_title('')
    opt_history_subplot.legend(loc='upper right', fontsize="16")
    opt_history = opt_history_subplot.figure
    opt_history.set_size_inches(10, 8)
    plt.tight_layout()


    ### param importance ###
    param_importance_subplot = plot_param_importances(study, target_name='Mean Absolute Error')
    param_importance_subplot.xaxis.label.set_fontweight('bold')
    param_importance_subplot.xaxis.label.set_fontsize(18)
    #param_importance_subplot.yaxis.set_label('Hyperparameters')
    param_importance_subplot.yaxis.label.set_fontweight('bold')
    param_importance_subplot.yaxis.label.set_fontsize(18)
    param_importance_subplot.tick_params(axis='x', labelsize=14)
    param_importance_subplot.tick_params(axis='y', labelsize=14)
    param_importance_subplot.set_title('')
    param_importance = param_importance_subplot.figure
    param_importance.set_size_inches(10, 8)
    plt.tight_layout()

    ### contours ###
    nbr_tanh_to_nbr_layers_subplot = plot_contour(study, params=["nbr_tanh", "nbr_layers"], target_name='Mean Absolute Error', )
    nbr_tanh_to_nbr_layers_subplot.xaxis.label.set_fontweight('bold')
    nbr_tanh_to_nbr_layers_subplot.yaxis.label.set_fontweight('bold')
    nbr_tanh_to_nbr_layers_subplot.set_title('')
    nbr_tanh_to_nbr_layers = nbr_tanh_to_nbr_layers_subplot.figure
    plt.tight_layout()

    nbr_tanh_to_batch_size_subplot = plot_contour(study, params=["nbr_tanh", "batch_size"], target_name='Mean Absolute Error', )
    nbr_tanh_to_batch_size_subplot.xaxis.label.set_fontweight('bold')
    nbr_tanh_to_batch_size_subplot.yaxis.label.set_fontweight('bold')
    nbr_tanh_to_batch_size_subplot.set_title('')
    nbr_tanh_to_batch_size = nbr_tanh_to_batch_size_subplot.figure
    plt.tight_layout()

    nbr_tanh_to_lr_subplot = plot_contour(study, params=["nbr_tanh", "lr"], target_name='Mean Absolute Error', )
    nbr_tanh_to_lr_subplot.xaxis.label.set_fontweight('bold')
    nbr_tanh_to_lr_subplot.yaxis.label.set_fontweight('bold')
    nbr_tanh_to_lr_subplot.set_title('')
    nbr_tanh_to_lr = nbr_tanh_to_lr_subplot.figure
    plt.tight_layout()

    lr_to_batch_size_subplot = plot_contour(study, params=["lr", "batch_size"], target_name='Mean Absolute Error', )
    lr_to_batch_size_subplot.xaxis.label.set_fontweight('bold')
    lr_to_batch_size_subplot.yaxis.label.set_fontweight('bold')
    lr_to_batch_size_subplot.set_title('')
    lr_to_batch_size = lr_to_batch_size_subplot.figure
    plt.tight_layout()

    batch_size_to_nbr_layers_subplot = plot_contour(study, params=["batch_size", "nbr_layers"], target_name='Mean Absolute Error', )
    batch_size_to_nbr_layers_subplot.xaxis.label.set_fontweight('bold')
    batch_size_to_nbr_layers_subplot.yaxis.label.set_fontweight('bold')
    batch_size_to_nbr_layers_subplot.set_title('')
    batch_size_to_nbr_layers = batch_size_to_nbr_layers_subplot.figure
    plt.tight_layout()

    lr_to_nbr_layers_subplot = plot_contour(study, params=["lr", "nbr_layers"], target_name='Mean Absolute Error', )
    lr_to_nbr_layers_subplot.xaxis.label.set_fontweight('bold')
    lr_to_nbr_layers_subplot.yaxis.label.set_fontweight('bold')
    lr_to_nbr_layers_subplot.set_title('')
    lr_to_nbr_layers = lr_to_nbr_layers_subplot.figure
    plt.tight_layout()

    ### slices ###
    slice_all_params_subplot = plot_parallel_coordinate(study, params=["lr", "batch_size", "nbr_layers", "nbr_tanh", "layer0_neurons", "layer1_neurons", "layer2_neurons", "layer3_neurons", "layer6_neurons", "layer7_neurons", "layer8_neurons"], target_name='Mean Absolute Error')
    slice_all_params_subplot.xaxis.label.set_fontweight('bold')
    slice_all_params_subplot.yaxis.label.set_fontweight('bold')
    slice_all_params = slice_all_params_subplot.figure
    slice_all_params.set_size_inches(20, 8)
    plt.tight_layout()

    slice_networks_structure_subplot = plot_parallel_coordinate(study, params=["layer0_neurons", "layer1_neurons", "layer2_neurons", "layer3_neurons", "layer6_neurons", "layer7_neurons", "layer8_neurons"], target_name='Mean Absolute Error')
    slice_networks_structure_subplot.xaxis.label.set_fontweight('bold')
    slice_networks_structure_subplot.yaxis.label.set_fontweight('bold')
    slice_networks_structure = slice_networks_structure_subplot.figure
    slice_networks_structure.set_size_inches(20, 8)
    plt.tight_layout()

    slice_training_params_subplot = plot_parallel_coordinate(study, params=["lr", "batch_size", "nbr_layers", "nbr_tanh"])
    slice_training_params_subplot.xaxis.label.set_fontweight('bold')
    slice_training_params_subplot.yaxis.label.set_fontweight('bold')
    slice_training_params = slice_training_params_subplot.figure
    slice_training_params.set_size_inches(20, 8)
    plt.tight_layout()

    # save images
    opt_history.savefig('./img/optimizer/ik/opt_history.png', format='png', dpi=150)
    param_importance.savefig('./img/optimizer/ik/param_importance.png', format='png', dpi=150)
    nbr_tanh_to_nbr_layers.savefig('./img/optimizer/ik/nbr_tanh_to_nbr_layers.png', format='png', dpi=150)
    nbr_tanh_to_batch_size.savefig('./img/optimizer/ik/nbr_tanh_to_batch_size.png', format='png', dpi=150)
    nbr_tanh_to_lr.savefig('./img/optimizer/ik/nbr_tanh_to_lr.png', format='png', dpi=150)
    lr_to_batch_size.savefig('./img/optimizer/ik/lr_to_batch_size.png', format='png', dpi=150)
    batch_size_to_nbr_layers.savefig('./img/optimizer/ik/batch_size_to_nbr_layers.png', format='png', dpi=150)
    lr_to_nbr_layers.savefig('./img/optimizer/ik/lr_to_nbr_layers.png', format='png', dpi=150)
    slice_all_params.savefig('./img/optimizer/ik/slice_all_params.png', format='png', dpi=150)
    slice_networks_structure.savefig('./img/optimizer/ik/slice_networks_structure.png', format='png', dpi=150)
    slice_training_params.savefig('./img/optimizer/ik/slice_training_params.png', format='png', dpi=150)

    print(study.best_trial)

if __name__ == '__main__':
    main()