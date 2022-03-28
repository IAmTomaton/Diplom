import math

import gym
from torch import nn

import DRQNERSolver
from DRQNAgent import DRQNAgent
from DRQNSTAgent import DRQNSTAgent
from DRQNSTCDAgent import DRQNSTCDAgent
from SequentialNetworkWithTypes import SequentialNetworkWithTypes, LayerType
from other.Noises import DiscreteUniformNoise
from other.SimpleControlProblem_Discrete import SimpleControlProblem_Discrete
from test2 import DRQN2Agent


def main():
    # env = gym.make("CartPole-v1")
    # env = DubinsCar()
    env = SimpleControlProblem_Discrete()

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    noise = DiscreteUniformNoise(action_n, threshold_decrease=0.002)
    network = SequentialNetworkWithTypes(state_dim,
                                         [(LayerType.Dense, 128),
                                          (LayerType.LSTM, 64),
                                          (LayerType.Dense, 64),
                                          (LayerType.Dense, action_n)],
                                         nn.ReLU())
    agent = DRQNAgent(state_dim, action_n, network, noise, batch_size=32, gamma=1, learning_rate=1e-2,
                      tau=1e-2, history_len=math.inf)

    DRQNERSolver.train(env, agent, epoch_n=1000, log=True, log_folder='logs\\SimpleControlProblem_Discrete',
                       name_suffix='inf_test_1')


if __name__ == '__main__':
    main()
