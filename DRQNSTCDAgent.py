import copy
import math
import random
import gym
import numpy as np
import torch
from torch import nn
from time import time
from Noise import DiscreteUniformNoise
from SequentialNetwork import SequentialNetwork, LayerType
from other.DubinsCar_Discrete import DubinsCar
from other.SimpleControlProblem_Discrete import SimpleControlProblem_Discrete
from train_info.train_log import TrainLog
from train_info.epoch_log import EpochLog
from log import save_log
from utils import print_log


class DRQNSTCDAgent:

    def __init__(self, network, noise, state_dim, action_n, gamma=1, memory_size=30000, batch_size=64,
                 states_count=4, learning_rate=1e-3, tau=1e-3):
        self._state_dim = state_dim
        self._action_n = action_n

        self.noise = noise

        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.states_count = states_count
        self.learning_rate = learning_rate
        self.tau = tau

        self._memory = []
        self._q = network
        self._q_target = copy.deepcopy(self._q)
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

    def get_action(self, states, train=False):
        if train and np.random.uniform(0, 1) < self.noise.threshold:
            return self.noise.get()

        memories = self.get_initial_state(1)

        for state in states:
            state_tensor = torch.FloatTensor(np.array([state]))

            if train:
                readouts, memories = self._q.step(state_tensor, memories)
            else:
                readouts, memories = self._q_target.step(state_tensor, memories)

        argmax_action = torch.argmax(readouts)
        return int(argmax_action)

    def fit_agent(self, state, action, reward, done, next_state):
        self._memory.append([state, action, reward, done, next_state])

        if len(self._memory) > self.memory_size:
            self._memory.pop(0)

        if len(self._memory) - self.states_count > self.batch_size:
            batch = self._get_batch()
            memories = self.get_initial_state(self.batch_size)
            next_memories = self.get_initial_state(self.batch_size)

            for k in range(self.states_count):
                states, actions, rewards, danes, next_states = list(zip(*batch[k]))

                states_tensor = torch.FloatTensor(np.array(states))
                q_values, memories = self._q(states_tensor, memories)

                next_states_tensor = torch.FloatTensor(np.array(next_states))
                next_q_values, next_memories = self._q_target(next_states_tensor, next_memories)

            targets = q_values.clone()
            for i in range(self.batch_size):
                targets[i][actions[i]] = rewards[i] + self.gamma * (1 - danes[i]) * max(next_q_values[i])

            loss = torch.mean((targets.detach() - q_values) ** 2)

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            for target_param, param in zip(self._q_target.parameters(), self._q.parameters()):
                target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

    def get_initial_state(self, batch_size):
        return self._q.get_initial_state(batch_size)

    def get_hyper_parameters(self):
        return {
            'agent_parameters': {
                'gamma': self.gamma,
                'memory_size': self.memory_size,
                'batch_size': self.batch_size,
                'states_count': self.states_count,
                'learning_rate': self.learning_rate,
                'tau': self.tau
            },
            'noise_parameters': self.noise.get_hyper_parameters(),
            'network_parameters': self._q.get_hyper_parameters(),
        }

    def _get_batch(self):
        batch_indexes = random.sample(range(self.states_count - 1, len(self._memory)), self.batch_size)
        batch = [[self._memory[j] for j in batch_indexes]]
        for i in range(1, self.states_count):
            batch.append([self._memory[j - i] if j - i >= 0 and not self._memory[j - i][3] else batch[i - 1][index]
                          for index, j in enumerate(batch_indexes)])
        batch.reverse()
        return batch


def get_session(agent, env, train_agent=False):
    state = env.reset()
    states = [state] * agent.states_count
    done = False

    total_reward = 0
    while not done:
        action = agent.get_action(states, train=train_agent)
        next_state, reward, done, _ = env.step(action)
        if train_agent:
            agent.fit_agent(state, action, reward, done, next_state)
        states.pop(0)
        states.append(next_state)
        state = next_state
        total_reward += reward

        if done:
            break

    return total_reward


def train(env, agent, log_folder='logs', name='DRQNSTCD', epoch_n=200, session_n=20, test_n=20):
    train_info = TrainLog(name, agent.get_hyper_parameters())

    for epoch in range(epoch_n):
        t = time()
        rewards = [get_session(agent, env, train_agent=True) for _ in range(session_n)]
        agent.noise.reduce()
        mean_reward = np.mean(rewards)

        test_rewards = [get_session(agent, env) for _ in range(test_n)]
        test_mean_reward = np.mean(test_rewards)
        std_dev = math.sqrt(np.mean([(r - test_mean_reward) ** 2 for r in test_rewards]))

        epoch_info = EpochLog(time() - t, mean_reward, rewards, test_mean_reward, test_rewards)
        train_info.add_epoch(epoch_info)

        save_log(train_info, log_folder + '\\' + train_info.name)
        print_log(epoch, mean_reward, time() - t, test_mean_reward, std_dev)


def main():
    env = gym.make("CartPole-v1")
    # env = DubinsCar()
    # env = SimpleControlProblem_Discrete()

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    noise = DiscreteUniformNoise(action_n)
    network = SequentialNetwork(state_dim,
                                [(LayerType.Dense, 128),
                                 (LayerType.LSTM, 64),
                                 (LayerType.Dense, 32),
                                 (LayerType.Dense, action_n)],
                                nn.ReLU())
    agent = DRQNSTCDAgent(network, noise, state_dim, action_n)

    train(env, agent, 'logs\\CartPole', 'DRQNSTCD')


if __name__ == '__main__':
    main()
