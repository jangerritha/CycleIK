# Copyright (c) 2023. Jan-Gerrit Habekost. GNU General Public License https://www.gnu.org/licenses/gpl-3.0.html.

import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

#plt.rcParams['font.size'] = 12
#plt.rcParams['font.weight'] = 'bold'
# Set the axes title font size
plt.rc('axes', labelsize=15)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=12)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=12)
# Set the legend font size
plt.rc('legend', fontsize=12)
# Set the font size of the figure title
plt.rc('figure', titlesize=12)
#matplotlib.rc('label', labelsize=20)
#matplotlib.rc('ytick', labelsize=20)

def main():
    robot = "nicol"

    ik_full_on_small =   [1.6215, 1.110,  0.6139, 0.7318, 1.0699, 1.0425, 6.0]
    ik_small_errors =   [1.3425, 0.6016,  0.4583, 0.3472, 0.2953, 0.2964, 0.2749]
    ik_full_errors =    [2.0238,  1.457,  1.0221,   1.172,   1.5190, 1.681, 6.0]
    ik_full_1mio_errors = [3.1720, 1.5167, 1.654, 1.7188, 1.3619, 6.0]
    gan_small_errors = [6.2028, 3.9612, 4.2949, 3.3177, 2.892, 20.0]
    gan_full_errors =   [4.0819, 4.5565, 3.5515, 3.2947, 2.7955, 20.0]
    gan_full_1mio_errors = [7.0762, 4.4462, 3.9986, 4.2755, 20.0, 20.0]
    time_series =       [10,  50,  100,  200,  300, 400, 500]
    time_series_GAN = [10, 20, 30, 40, 50, 60]

    rot_ik_full_on_small = [0.1715, 0.0952, 0.0583, 0.0722, 0.1250, 0.1173, 1.0]
    rot_ik_small_errors = [0.1665, 0.0621, 0.0473, 0.0381, 0.0335, 0.0345, 0.0328]
    rot_ik_full_errors = [0.274, 0.3359, 0.1523, 0.1503, 0.4008, 0.3214, 1.0]
    rot_ik_full_1mio_errors = [0.4616, 0.2918, 0.248, 0.3542, 0.2616, 1.0]
    rot_gan_small_errors = [0.6659, 0.3803, 0.3963, 0.2805, 0.2662, 4.0]
    rot_gan_full_errors = [0.5079, 0.7302, 0.5737, 0.6564, 0.5631, 4.0]
    rot_gan_full_1mio_errors = [0.9354, 0.7674, 0.6493, 0.4917, 4.0, 4.0]

    fig, ax = plt.subplots(2, 2, figsize=(16,9))
    #print(ax)
    ax[0][0].set_xlabel('Training Epochs')
    ax[0][0].set_ylabel('Error (mm)')
    kinematic_loss = ax[0][0].plot(time_series, ik_small_errors, marker="^", color="royalblue")
    kinematic_loss[0].set_label(r'MLP - Training: $Small_{1000}$ - Test: $Small_{100}$', )
    kinematic_loss = ax[0][0].plot(time_series, ik_full_errors, marker="^", color="forestgreen")
    kinematic_loss[0].set_label(r'MLP - Training: $Full_{1400}$ - Test: $Full_{140}$')
    kinematic_loss = ax[0][0].plot(time_series[:6], ik_full_1mio_errors, marker="^", color="saddlebrown")
    kinematic_loss[0].set_label(r'MLP - Training: $Full_{1000}$ - Test: $Full_{100}$')
    kinematic_loss = ax[0][0].plot(time_series, ik_full_on_small, linestyle='dashed', marker="^", color="forestgreen")
    kinematic_loss[0].set_label(r'MLP - Training: $Full_{1400}$ - Test: $Small_{100}$')
    ax[0][0].legend(loc='upper center', bbox_to_anchor=(0.35, 1.15))
    ax[0][0].set_yticks(np.arange(0.0, 3.5, 0.5))
    x_axis_scale = np.arange(0, 600, 100)
    x_axis_scale[0] = 10
    print(x_axis_scale)
    ax[0][0].set_xticks(x_axis_scale)
    ax[0][0].set_ylim(top=3.5, bottom=0.0)


    ax[1][0].set_xlabel('Training Epochs')
    ax[1][0].set_ylabel('Error (mm)')
    kinematic_loss = ax[1][0].plot(time_series_GAN, gan_small_errors, marker="^", color="orangered")
    kinematic_loss[0].set_label(r'GAN - Training: $Small_{1000}$ - Test: $Small_{100}$')
    kinematic_loss = ax[1][0].plot(time_series_GAN, gan_full_errors, marker="^", color="purple")
    kinematic_loss[0].set_label(r'GAN - Training: $Full_{1400}$ - Test: $Full_{140}$')
    kinematic_loss = ax[1][0].plot(time_series_GAN, gan_full_1mio_errors, marker="^", color="magenta")
    kinematic_loss[0].set_label(r'GAN - Training: $Full_{1000}$ - Test: $Full_{100}$')
    ax[1][0].legend(loc='upper center', bbox_to_anchor=(0.4, 1.1))
    ax[1][0].set_yticks(np.arange(0.0, 10.5, 1.0))
    x_axis_scale = np.arange(0, 60, 10)
    x_axis_scale[0] = 10
    print(x_axis_scale)
    ax[1][0].set_xticks(x_axis_scale)
    ax[1][0].set_ylim(top=10.5)
    ax[1][0].set_xlim(right=55)

    # print(ax)
    ax[0][1].set_xlabel('Training Epochs')
    ax[0][1].set_ylabel('Error (degree)')
    kinematic_loss = ax[0][1].plot(time_series, rot_ik_small_errors, marker="D", color="royalblue")
    kinematic_loss[0].set_label(r'MLP - Training: $Small_{1000}$ - Test: $Small_{100}$', )
    kinematic_loss = ax[0][1].plot(time_series, rot_ik_full_errors, marker="D", color="forestgreen")
    kinematic_loss[0].set_label(r'MLP - Training: $Full_{1400}$ - Test: $Full_{140}$')
    kinematic_loss = ax[0][1].plot(time_series[:6], rot_ik_full_1mio_errors, marker="D", color="saddlebrown")
    kinematic_loss[0].set_label(r'MLP - Training: $Full_{1000}$ - Test: $Full_{100}$')
    kinematic_loss = ax[0][1].plot(time_series, rot_ik_full_on_small, linestyle='dashed', marker="D", color="forestgreen")
    kinematic_loss[0].set_label(r'MLP - Training: $Full_{1400}$ - Test: $Small_{100}$')
    ax[0][1].legend(loc='upper center', bbox_to_anchor=(0.35, 1.15))
    ax[0][1].set_yticks(np.arange(0.0, 0.6, 0.1))
    x_axis_scale = np.arange(0, 600, 100)
    x_axis_scale[0] = 10
    print(x_axis_scale)
    ax[0][1].set_xticks(x_axis_scale)
    ax[0][1].set_ylim(top=0.6, bottom=0.0)

    ax[1][1].set_xlabel('Training Epochs')
    ax[1][1].set_ylabel('Error (degree)')
    kinematic_loss = ax[1][1].plot(time_series_GAN, rot_gan_small_errors, marker="D", color="orangered")
    kinematic_loss[0].set_label(r'GAN - Training: $Small_{1000}$ - Test: $Small_{100}$')
    kinematic_loss = ax[1][1].plot(time_series_GAN, rot_gan_full_errors, marker="D", color="purple")
    kinematic_loss[0].set_label(r'GAN - Training: $Full_{1400}$ - Test: $Full_{140}$')
    kinematic_loss = ax[1][1].plot(time_series_GAN, rot_gan_full_1mio_errors, marker="D", color="magenta")
    kinematic_loss[0].set_label(r'GAN - Training: $Full_{1000}$ - Test: $Full_{100}$')
    ax[1][1].legend(loc='upper center', bbox_to_anchor=(0.4, 1.1))
    ax[1][1].set_yticks(np.arange(0.0, 1.6, 0.2))
    x_axis_scale = np.arange(0, 60, 10)
    x_axis_scale[0] = 10
    print(x_axis_scale)
    ax[1][1].set_xticks(x_axis_scale)
    ax[1][1].set_ylim(top=1.6)
    ax[1][1].set_xlim(right=55)

    fig.tight_layout()
    plt.savefig('./img/losses/losses_for_diff_epochs.png', format='png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()