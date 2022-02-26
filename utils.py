import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(28, 14))


def show_plot(total_rewards):
    plt.plot(total_rewards)
    plt.ylabel('reward')
    plt.show()


def save_plot(total_rewards, total_test_rewards, total_test_mean, total_test_std_dev, path, mean_range=20):
    fig.clf()

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(total_rewards)
    ax1.set_ylabel('reward')

    ax2 = fig.add_subplot(2, 2, 3)
    mean = [np.mean(total_rewards[i:i + mean_range]) for i in range(len(total_rewards) - mean_range)]
    ax2.plot(mean)
    ax2.set_ylabel('mean')

    ax3 = fig.add_subplot(2, 2, 2)
    ax3.plot(total_test_rewards)
    ax3.set_ylabel('test reward')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(total_test_mean)
    ax4.plot(total_test_std_dev)
    ax4.set_ylabel('test mean/std dev')

    fig.savefig(path)


def print_log(epoch, mean_reward, time, epsilon, test_mean_reward, std_dev):
    text = 'Epoch: ' + str(epoch) + ' Mean reward: ' + str(round(mean_reward, 2)) + \
           ' Time: ' + str(round(time, 2)) + ' Ep: ' + str(round(epsilon, 5)) + \
           ' Test: ' + str(round(test_mean_reward, 2)) + ' Std Dev: ' + str(round(std_dev, 2))

    print(text)
