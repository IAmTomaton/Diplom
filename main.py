import math
from time import time

import gym
import numpy as np
from torch import nn

import DRQNERSolver
from DRQNAgent import DRQNAgent
from DRQNFinalHistoryAgent import DRQNFinalHistoryAgent
from DRQNSTAgent import DRQNSTAgent
from DRQNSTCDAgent import DRQNSTCDAgent
from SequentialNetworkWithTypes import SequentialNetworkWithTypes, LayerType
from log import save_log
from other.Noises import DiscreteUniformNoise
from other.SimpleControlProblem_Discrete import SimpleControlProblem_Discrete
from test2 import DRQN2Agent
from train_info.epoch_log import EpochLog
from train_info.train_log import TrainLog
from utils import print_log


def add_session(sessions, session):
    sessions += session


def train(env, agent, epoch_n=200, episode_n=10, test_episode_n=10, log=False, log_folder='', name_suffix=''):
    if log:
        train_info = TrainLog(agent.name + name_suffix, agent.get_hyper_parameters())

    for epoch in range(epoch_n):
        sessions = []

        start_time = time()
        DRQNERSolver.go(env, agent, episode_n, show=lambda a, b, c, session: add_session(sessions, session))
        epoch_time = time() - start_time

        if log:
            rewards = [sum(session['rewards']) for session in sessions]
            mean_reward = np.mean(rewards)

            agent.noise.turn_off()
            test_sessions = DRQNERSolver.go(env, agent, test_episode_n, False)
            agent.noise.turn_on()

            test_rewards = [sum(session['rewards']) for session in test_sessions]
            test_mean_reward = np.mean(test_rewards)
            std_dev = math.sqrt(np.mean([(r - test_mean_reward) ** 2 for r in test_rewards]))

            epoch_info = EpochLog(epoch_time, mean_reward, rewards, test_mean_reward, test_rewards)
            train_info.add_epoch(epoch_info)

            save_log(train_info, log_folder + '\\' + train_info.name)
            print_log(epoch, mean_reward, epoch_time, test_mean_reward, std_dev)


def main():
    env = gym.make("CartPole-v1")
    # env = DubinsCar()
    # env = SimpleControlProblem_Discrete()

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    noise = DiscreteUniformNoise(action_n, threshold_decrease=0.002)
    network = SequentialNetworkWithTypes(state_dim,
                                         [(LayerType.Dense, 128),
                                          (LayerType.LSTM, 64),
                                          (LayerType.Dense, 64),
                                          (LayerType.Dense, action_n)],
                                         nn.ReLU())
    agent = DRQNFinalHistoryAgent(state_dim, action_n, network, noise, batch_size=128, gamma=1, learning_rate=1e-2,
                                  tau=1e-2, history_len=4)

    train(env, agent, epoch_n=1000, log=True, log_folder='logs\\CartPole',
                       name_suffix='_test_1')


if __name__ == '__main__':
    main()
