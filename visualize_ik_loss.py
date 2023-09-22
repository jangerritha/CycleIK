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
    robot = "nicol"
    #data = pickle.load(open(f"./weights/nicol/TRAINING_SAVES/100_epochs/1400/train_ik_loss_with_kinematics.p", 'rb'))
    data = pickle.load(open(f"/home/jan-gerrit/repositories/cycleik/results/train_ik_loss_1000.p", 'rb'))
    time_series = []
    small_time_series = [1]
    print(len(data))
    print(data[0])
    print(len(data[0]))
    print(data[1])
    print(len(data[1]))
    for i in range(len(data[0])):
        #if i == 0: continue
        time_series.append(i)

    for i in range(int(len(data[0]) / 10)):
        #if i == 0: continue
        small_time_series.append((i+1) * 10)

    print(len(time_series))

    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    kinematic_loss = ax.plot(time_series, data[0][:])
    #kinematic_loss = ax.plot(time_series, data[0])
    kinematic_loss[0].set_label('Kinematic Loss (MAE)')
    kinematic_loss = ax.plot(time_series, data[2][:])
    #kinematic_loss = ax.plot(small_time_series, data[2])
    kinematic_loss[0].set_label('Validation Loss (MAE)')
    # kinematic_loss = ax.plot(time_series, data[1][1:])
    #kinematic_loss = ax.plot(time_series, data[1])
    #kinematic_loss[0].set_label('Zero-Controller Loss (MSE)')
    ax.legend(loc='upper right')#, bbox_to_anchor=(1.1, 0.5))
    #ax.legend(loc='lower right', bbox_to_anchor=(0.35, -1.15))
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    #ax.legend()
    plt.yticks(np.arange(0.0, 0.025, 0.002))
    #plt.yticks(np.arange(0.0, 0.2, 0.02))
    plt.annotate('train: %0.5f' % data[0][-1], xy=(1, data[0][-1]), xytext=(-81, 22),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.annotate('val:    %0.5f' % data[2][-1], xy=(1, data[2][-1]), xytext=(-81, 14),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('./img/losses/vis_ik_loss_zoom.png')
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    #kinematic_loss = ax.plot(time_series, data[0][:])
    kinematic_loss = ax.plot(time_series, data[0])
    kinematic_loss[0].set_label('Kinematic Loss (MAE)')
    #kinematic_loss = ax.plot(small_time_series, data[2][:])
    kinematic_loss = ax.plot(time_series, data[2])
    kinematic_loss[0].set_label('Validation Loss (MAE)')
    # kinematic_loss = ax.plot(time_series, data[1][1:])
    kinematic_loss = ax.plot(time_series, data[1])
    kinematic_loss[0].set_label('Zero-Controller Loss (MSE)')
    ax.legend(loc='center right')  # , bbox_to_anchor=(1.1, 0.5))
    # ax.legend(loc='lower right', bbox_to_anchor=(0.35, -1.15))
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    # ax.legend()
    #plt.yticks(np.arange(0.0, 0.025, 0.002))
    plt.yticks(np.arange(0.0, 0.2, 0.02))
    plt.annotate('train: %0.5f' % data[0][-1], xy=(1, data[0][-1]), xytext=(-81, 21),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.annotate('val:    %0.5f' % data[2][-1], xy=(1, data[2][-1]), xytext=(-81, 8),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('./img/losses/vis_ik_loss.png')
    plt.show()

if __name__ == '__main__':
    main()