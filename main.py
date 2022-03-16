import gym
from torch import nn

import DRQNERSolver
from DRQNSTAgent import DRQNSTAgent
from SequentialNetwork import SequentialNetwork, LayerType
from other.Noises import DiscreteUniformNoise


def main():
    env = gym.make("CartPole-v1")
    # env = DubinsCar()
    # env = SimpleControlProblem_Discrete()

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    noise = DiscreteUniformNoise(action_n, threshold_decrease=0.01)
    network = SequentialNetwork(state_dim,
                                [(LayerType.Dense, 2),
                                 (LayerType.LSTM, 4),
                                 (LayerType.Dense, 32),
                                 (LayerType.Dense, action_n)],
                                nn.ReLU())
    agent = DRQNSTAgent(network, noise, state_dim, action_n, batch_size=3, gamma=1, learning_rate=1e-3,
                          tau=1e-3)

    DRQNERSolver.train(env, agent, 'logs\\CartPole', 'DRQNST_test')


if __name__ == '__main__':
    main()
