import matplotlib.pyplot as plt
import pickle
import numpy as np

def main():
    data = pickle.load(open("./results/train_fk_loss.p", 'rb'))
    time_series = []
    print(len(data))
    print(data[0])
    print(len(data[0]))
    print(data[1])
    print(len(data[1]))
    for i in range(len(data[0])):
        #if i == 0: continue
        time_series.append(i * 10)

    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    kinematic_loss = ax.plot(time_series, data[0])
    kinematic_loss[0].set_label('Kinematic Loss (MAE)')
    kinematic_loss = ax.plot(time_series, data[1])
    kinematic_loss[0].set_label('Validation Loss (MAE)')
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.legend()
    plt.yticks(np.arange(0.0, 0.2, 0.01))
    plt.annotate('train: %0.5f' % data[0][-1], xy=(1, data[0][-1]), xytext=(-80, 18),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.annotate('val:    %0.5f' % data[1][-1], xy=(1, data[1][-1]), xytext=(-81, 5),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.savefig('./img/losses/vis_fk_loss.png')
    plt.show()


if __name__ == '__main__':
    main()