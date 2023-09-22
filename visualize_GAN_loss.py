# Copyright (c) 2023. Jan-Gerrit Habekost. GNU General Public License https://www.gnu.org/licenses/gpl-3.0.html.

import matplotlib.pyplot as plt
import pickle
import numpy as np

# Set the axes title font size
plt.rc('axes', labelsize=14)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=12)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=12)
# Set the legend font size
plt.rc('legend', fontsize=12)
# Set the font size of the figure title
plt.rc('figure', titlesize=12)

def main():
    data = pickle.load(open("./weights/nicol/TRAINING_SAVES/50_epochs/1400/train_GAN_with_kinematics_loss.p", 'rb'))
    #data = pickle.load(open("./results/nicol/train_GAN_with_kinematics_loss.p", 'rb'))
    time_series = [1]
    last_step = 0
    time_series_small = []
    print(len(data))
    print(data[0])
    print(len(data[0]))
    print(data[1])
    print(len(data[1]))
    for i in range(len(data[2])):
        if i == 0: continue
        last_step = last_step + 10
        time_series.append(last_step)


    print(time_series)
    for i in range(len(data[0])):
        time_series_small.append(i + 1)

    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    # kinematic_loss = ax.plot(time_series, data[0][1:])
    kinematic_loss = ax.plot(time_series_small, data[0])
    kinematic_loss[0].set_label('Kinematic Loss (MAE)')
    # kinematic_loss = ax.plot(time_series, data[1][1:])
    kinematic_loss = ax.plot(time_series, data[2])
    kinematic_loss[0].set_label('Validation Loss (MAE)')
    kinematic_loss = ax.plot(time_series_small, data[1])
    kinematic_loss[0].set_label('Variance Loss (MSE)')
    # kinematic_loss = ax.plot(time_series, data[2][1:])
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.legend()
    plt.yticks(np.arange(0.0, 0.05, 0.005))
    plt.annotate('train: %0.5f' % data[0][-1], xy=(1, data[0][-1]), xytext=(-81, 23),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.annotate('val:    %0.5f' % data[2][-1], xy=(1, data[2][-1]), xytext=(-81, 11),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('./img/losses/vis_GAN_loss.png')
    plt.show()


if __name__ == '__main__':
    main()