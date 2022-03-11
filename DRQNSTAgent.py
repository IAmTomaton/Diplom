import copy
import math
import gym
import numpy as np
import torch
from torch import nn
from time import time
from Noise import DiscreteUniformNoise
from SequentialNetwork import SequentialNetwork, LayerType
from train_info.train_log import TrainLog
from Buffer import Buffer
from train_info.epoch_log import EpochLog
from log import save_log
from other.SimpleControlProblem_Discrete import SimpleControlProblem_Discrete
from other.DubinsCar_Discrete import DubinsCar
from utils import print_log


class DRQNSTAgent:

    def __init__(self, network, noise, state_dim, action_n, gamma=1, memory_size=30000, batch_size=32,
                 burn_in=8, batch_len=12, learning_rate=1e-3, tau=1e-3):
        self._state_dim = state_dim
        self._action_n = action_n

        self.noise = noise

        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.burn_in = burn_in
        self.batch_len = batch_len
        self.learning_rate = learning_rate
        self.tau = tau

        self._memory = Buffer(self.memory_size)
        self._q = network
        self._q_target = copy.deepcopy(self._q)
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

    def get_action(self, state, prev_memories, train=False):
        state = torch.FloatTensor(np.array([state]))

        if train:
            readouts, new_memories = self._q.step(state, prev_memories)
            if np.random.uniform(0, 1) < self.noise.threshold:
                return self.noise.get(), new_memories
        else:
            readouts, new_memories = self._q_target.step(state, prev_memories)

        argmax_action = torch.argmax(readouts)
        return int(argmax_action), new_memories

    def fit_agent(self, state, action, reward, done, next_state):
        self._memory.add([state, action, reward, done, next_state])

        if len(self._memory) > self.batch_size * (self.batch_len - self.burn_in):
            batch = self._memory.get_batch(self.batch_size, self.batch_len)
            memories = self.get_initial_state(self.batch_size)
            loss = 0

            for k in range(self.batch_len):
                states, actions, rewards, danes, next_states = list(zip(*batch[k]))

                states = torch.FloatTensor(np.array(states))
                q_values, lstm_states = self._q(states, memories)

                next_states = torch.FloatTensor(np.array(next_states))
                next_q_values, next_lstm_states = self._q_target(next_states, lstm_states)
                memories = lstm_states

                m = torch.zeros((self.batch_size, self.batch_size))
                for i in range(self.batch_size):
                    if not danes[i]:
                        m[i][i] = 1
                memories = torch.mm(m, memories[0]), torch.mm(m, memories[1])

                if k == self.burn_in:
                    h, c = memories
                    h.detach()
                    c.detach()

                if k >= self.burn_in:
                    targets = q_values.clone()
                    for i in range(self.batch_size):
                        targets[i][actions[i]] = rewards[i] + self.gamma * (1 - danes[i]) * max(next_q_values[i])

                    loss += torch.mean((targets.detach() - q_values) ** 2)

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
                'burn_in': self.burn_in,
                'batch_len': self.batch_len,
                'learning_rate': self.learning_rate,
                'tau': self.tau
            },
            'noise_parameters': self.noise.get_hyper_parameters(),
            'network_parameters': self._q.get_hyper_parameters(),
        }


def get_session(agent, env, batch_size=1, train_agent=False):
    state = env.reset()
    memory = agent.get_initial_state(batch_size)
    total_reward = 0
    while True:
        action, memory = agent.get_action(state, memory, train=train_agent)
        next_state, reward, done, _ = env.step(action)
        if train_agent:
            agent.fit_agent(state, action, reward, done, next_state)
        state = next_state
        total_reward += reward

        if done:
            break

    return total_reward


def train(env, agent, log_folder='logs', name='DRQNST', epoch_n=100, session_n=20, test_n=20):
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
    agent = DRQNSTAgent(network, noise, state_dim, action_n)

    train(env, agent, 'logs', 'test')


if __name__ == '__main__':
    main()
