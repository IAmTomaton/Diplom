import copy
import random
import numpy as np
import torch


class DRQNSTCDAgent:

    def __init__(self, network, noise, state_dim, action_n, gamma=1, memory_size=30000, batch_size=64, states_count=4,
                 learning_rate=1e-3, tau=1e-3):
        self._state_dim = state_dim
        self._action_n = action_n

        self.noise = noise

        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.states_count = states_count
        self.learning_rate = learning_rate
        self.tau = tau

        self._buffer = []
        self._q = network
        self._q_target = copy.deepcopy(self._q)
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

        self._prev_states = []

    def get_action(self, state, train=False):
        if not self._prev_states:
            self._prev_states = [state] * self.states_count
        self._prev_states.pop(0)
        self._prev_states.append(state)

        if train and np.random.uniform(0, 1) < self.noise.threshold:
            return self.noise.get()

        memories = self.get_initial_state(1)

        for state in self._prev_states:
            state_tensor = torch.FloatTensor(np.array([state]))

            if train:
                readouts, memories = self._q.step(state_tensor, memories)
            else:
                readouts, memories = self._q_target.step(state_tensor, memories)

        argmax_action = torch.argmax(readouts)
        return int(argmax_action)

    def fit_agent(self, state, action, reward, done, next_state):
        self._add_to_buffer([state, action, reward, done, next_state])

        if len(self._buffer) - self.states_count > self.batch_size:
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
                target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

    def get_initial_state(self, batch_size):
        return self._q.get_initial_state(batch_size)

    def reset(self):
        self._prev_states = []

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
        batch_indexes = random.sample(range(self.states_count - 1, len(self._buffer)), self.batch_size)
        batch = [[self._buffer[j] for j in batch_indexes]]
        for i in range(1, self.states_count):
            batch.append([self._buffer[j - i] if j - i >= 0 and not self._buffer[j - i][3] else batch[i - 1][index]
                          for index, j in enumerate(batch_indexes)])
        batch.reverse()
        return batch

    def _add_to_buffer(self, data):
        self._buffer.append(data)

        if len(self._buffer) > self.memory_size:
            self._buffer.pop(0)
