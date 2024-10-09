#  Copyright (c) 2024. Jan-Gerrit Habekost. GNU General Public License. https://www.gnu.org/licenses/gpl-3.0.html.


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
    robots = ["nicol", "nico", "fetch", "valkyrie", "panda"]
    robots_CAP = ["NICOL", "NICO", "Fetch", "Valkyrie", "Panda"]
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    #data = pickle.load(open(f"./weights/nicol/TRAINING_SAVES/100_epochs/1400/train_ik_loss_with_kinematics.p", 'rb'))
    all_data = {}
    for robot in robots:
        #data_runs = []

        all_pos_errors = np.empty((0,10))
        all_orientation_errors = np.empty((0, 10))
        for run in range(10):
            current_run = pickle.load(open(f"/home/jan-gerrit/repositories/cycleik_2/weights/IROS_saves_precision_26_02/results/{robot}/train_MLP_with_kinematics_loss_IROS_train_run_{run}.p", 'rb'))
            position = current_run[3]
            orientation = current_run[4]
            all_pos_errors = np.concatenate((all_pos_errors, np.array(position).reshape((1,10))), axis=0)
            all_orientation_errors = np.concatenate((all_orientation_errors, np.array(orientation).reshape((1,10))), axis=0)
            #data_runs.append([position, orientation])
        min_error_pos = np.min(all_pos_errors, axis=0)
        max_error_pos = np.max(all_pos_errors, axis=0)
        avg_error_pos = np.mean(all_pos_errors, axis=0)
        min_error_rot = np.min(all_orientation_errors, axis=0)
        max_error_rot = np.max(all_orientation_errors, axis=0)
        avg_error_rot = np.mean(all_orientation_errors, axis=0)
        error_dict = {}
        error_dict['pos'] = [avg_error_pos, min_error_pos, max_error_pos]
        error_dict['rot'] = [avg_error_rot, min_error_rot, max_error_rot]
        all_data[robot] = error_dict

        lowest_err = 10
        lowest_index = -1
        pos_error = all_pos_errors[:,9:]
        rot_error = all_orientation_errors[:,9:]
        all_err = (pos_error * 10) + rot_error
        for i in range(10):
            if all_err[i] < lowest_err:
                lowest_err = all_err[i]
                lowest_index = i

        print(f'{robot} best run: {lowest_index}, with error: {lowest_err}')


    time_series = []
    small_time_series = [1]
    print(len(all_data['nicol']['pos'][0]))
    print(all_data['nicol']['pos'][0])

    for i in range(len(all_data['nicol']['pos'][0])):
        #if i == 0: continue
        time_series.append(i)

    for i in range(int(len(all_data['nicol']['pos'][0]) / 10)):
        #if i == 0: continue
        small_time_series.append((i+1) * 10)

    print(len(time_series))

    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pos. Loss')
    #kinematic_loss = ax.plot(time_series, data[0][:])
    for e, robot in enumerate(robots):
        kinematic_loss = ax.plot(time_series, all_data[robot]['pos'][0])
        poly = ax.fill_between(time_series, all_data[robot]['pos'][1], all_data[robot]['pos'][2], facecolor=colors[e], alpha=0.5,interpolate=True)
        kinematic_loss[0].set_label(robots_CAP[e])
    #kinematic_loss = ax.plot(small_time_series, data[2][:])
    #kinematic_loss = ax.plot(time_series, data[4])
    #kinematic_loss[0].set_label('Validation Loss (MAE)')
    # kinematic_loss = ax.plot(time_series, data[1][1:])
    #kinematic_loss = ax.plot(time_series, data[1])
    #kinematic_loss[0].set_label('Zero-Controller Loss (MSE)')
    ax.legend(loc='upper right')  # , bbox_to_anchor=(1.1, 0.5))
    # ax.legend(loc='lower right', bbox_to_anchor=(0.35, -1.15))
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    # ax.legend()
    #plt.yticks(np.arange(0.0, 0.025, 0.002))
    plt.yticks(np.arange(0.0, 0.04, 0.005))
    #plt.annotate('train: %0.5f' % data[3][-1], xy=(1, data[3][-1]), xytext=(-81, 21),
    #             xycoords=('axes fraction', 'data'), textcoords='offset points')
    #plt.annotate('val:    %0.5f' % data[4][-1], xy=(1, data[4][-1]), xytext=(-81, 8),
    #             xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('./img/losses/vis_multi_run_loss.png')
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rot. Loss')
    # kinematic_loss = ax.plot(time_series, data[0][:])
    for e, robot in enumerate(robots):
        kinematic_loss = ax.plot(time_series, all_data[robot]['rot'][0])
        poly = ax.fill_between(time_series, all_data[robot]['rot'][1], all_data[robot]['rot'][2], facecolor=colors[e],
                               alpha=0.5, interpolate=True)
        kinematic_loss[0].set_label(robots_CAP[e])
    # kinematic_loss = ax.plot(small_time_series, data[2][:])
    # kinematic_loss = ax.plot(time_series, data[4])
    # kinematic_loss[0].set_label('Validation Loss (MAE)')
    # kinematic_loss = ax.plot(time_series, data[1][1:])
    # kinematic_loss = ax.plot(time_series, data[1])
    # kinematic_loss[0].set_label('Zero-Controller Loss (MSE)')
    ax.legend(loc='upper right')  # , bbox_to_anchor=(1.1, 0.5))
    # ax.legend(loc='lower right', bbox_to_anchor=(0.35, -1.15))
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    # ax.legend()
    # plt.yticks(np.arange(0.0, 0.025, 0.002))
    plt.yticks(np.arange(0.0, 0.3, 0.03))
    # plt.annotate('train: %0.5f' % data[3][-1], xy=(1, data[3][-1]), xytext=(-81, 21),
    #             xycoords=('axes fraction', 'data'), textcoords='offset points')
    # plt.annotate('val:    %0.5f' % data[4][-1], xy=(1, data[4][-1]), xytext=(-81, 8),
    #             xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('./img/losses/vis_multi_run_loss.png')
    plt.show()

if __name__ == '__main__':
    main()