from time import time
import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
from DRQNAgent import DRQNAgent, show_simulation, get_session

fig = plt.figure(figsize=(12, 7))


def show_plot(total_rewards):
    plt.plot(total_rewards)
    plt.ylabel('reward')
    plt.show()


def save_plot(total_rewards, total_test_rewards, total_test_mean, path, mean_range=10):
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
    ax4.set_ylabel('test mean')

    fig.savefig(path)


def train(hyper_parameters, device):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    agent = DRQNAgent(state_dim, action_n, hyper_parameters, device)

    total_rewards = []
    total_test_rewards = []
    total_test_mean = [0]
    epoch_n = 20
    session_n = 20
    test_n = 100
    t = time()

    for epoch in range(epoch_n):
        rewards = [get_session(agent, env, train_agent=True) for _ in range(session_n)]
        total_rewards += rewards
        mean_reward = np.mean(rewards)

        test_rewards = [get_session(agent, env) for _ in range(test_n)]
        total_test_rewards += test_rewards
        test_mean_reward = np.mean(test_rewards)
        total_test_mean.append(test_mean_reward)

        save_plot(total_rewards, total_test_rewards, total_test_mean, 'plots\\' + 'plot_lstm' + '.png')
        text = 'Epoch: ' + str(epoch + 1) + ' Mean reward: ' + str(mean_reward) + \
               ' Time: ' + str(round(time() - t, 2)) + ' Ep: ' + str(round(agent.epsilon, 5)) + \
               ' Test: ' + str(test_mean_reward)

        print(text)

        t = time()


def main():
    use_cuda = torch.cuda.is_available() and False
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Used', device)

    hyper_parameters = {'memory_size': 30000, 'gamma': 0.95, 'batch_size': 32, 'learning_rate': 1e-4,
                        'min_epsilon': 1e-3, 'mul_epsilon': 0.9999}

    train(hyper_parameters, device)


if __name__ == '__main__':
    main()
