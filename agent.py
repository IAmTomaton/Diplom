import numpy as np
import torch
from torch import nn
import random
from collections import deque


class Network(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, 64)
        self.linear_2 = nn.Linear(64, 64)
        self.linear_3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden = self.linear_1(x)
        hidden = self.relu(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.relu(hidden)
        output = self.linear_3(hidden)
        return output


class DQNAgent(nn.Module):

    def __init__(self, state_dim, action_n, hyper_parameters, device):
        super().__init__()
        self._state_dim = state_dim
        self._action_n = action_n
        self._device = device

        self.gamma = hyper_parameters['gamma'] if 'gamma' in hyper_parameters else 0.95
        self.epsilon = 1
        self.min_epsilon = hyper_parameters['min_epsilon'] if 'min_epsilon' in hyper_parameters else 0.001
        self.mul_epsilon = hyper_parameters['mul_epsilon'] if 'mul_epsilon' in hyper_parameters else 0.99
        self.memory_size = hyper_parameters['memory_size'] if 'memory_size' in hyper_parameters else 10000
        self.batch_size = hyper_parameters['batch_size'] if 'batch_size' in hyper_parameters else 64
        self.learning_rate = hyper_parameters['learning_rate'] if 'learning_rate' in hyper_parameters else 1e-2
        self.hyper_parameters = hyper_parameters

        self._memory = deque()
        self._q = Network(self._state_dim, self._action_n).to(device)
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

    def get_action(self, state, train=False):
        state = torch.FloatTensor(np.array(state)).to(device=self._device, non_blocking=True)
        argmax_action = torch.argmax(self._q(state))
        if not train:
            return int(argmax_action)
        probs = np.ones(self._action_n) * self.epsilon / self._action_n
        probs[argmax_action] += 1 - self.epsilon
        actions = np.arange(self._action_n)
        return np.random.choice(actions, p=probs)

    def fit_DQN(self, state, action, reward, done, next_state):
        self._memory.append([state, action, reward, done, next_state])
        if len(self._memory) > self.memory_size:
            self._memory.popleft()

        if len(self._memory) > self.batch_size:
            batch = random.sample(self._memory, self.batch_size)

            states, actions, rewards, danes, next_states = list(zip(*batch))
            states = torch.FloatTensor(np.array(states)).to(device=self._device, non_blocking=True)
            q_values = self._q(states)
            next_states = torch.FloatTensor(np.array(next_states)).to(device=self._device, non_blocking=True)
            next_q_values = self._q(next_states)
            targets = q_values.clone()
            for i in range(self.batch_size):
                targets[i][actions[i]] = rewards[i] + self.gamma * (1 - danes[i]) * max(next_q_values[i])

            loss = torch.mean((targets.detach() - q_values) ** 2)

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            self.epsilon = max(self.min_epsilon, self.epsilon * self.mul_epsilon)
