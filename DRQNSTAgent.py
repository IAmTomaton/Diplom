import copy
import random
import numpy as np
import torch


class DRQNSTAgent:

    def __init__(self, network, noise, state_dim, action_n, gamma=1, memory_size=30000, batch_size=32, burn_in=8,
                 batch_len=12, learning_rate=1e-3, tau=1e-3):
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

        self._buffer = []
        self._q = network
        self._q_target = copy.deepcopy(self._q)
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

        self._memory = self.get_initial_state(1)

    def get_action(self, state, train=False):
        state = torch.FloatTensor(np.array([state]))

        if train:
            readouts, self._memory = self._q.step(state, self._memory)
            if np.random.uniform(0, 1) < self.noise.threshold:
                return self.noise.get()
        else:
            readouts, self._memory = self._q_target.step(state, self._memory)

        argmax_action = torch.argmax(readouts)
        return int(argmax_action)

    def fit_agent(self, state, action, reward, done, next_state):
        self._add_to_buffer([state, action, reward, done, next_state])

        if len(self._buffer) > self.batch_size * self.batch_len:
            batch = self._get_batch()
            memories = self.get_initial_state(self.batch_size)
            loss = 0

            for k in range(self.batch_len):
                states, actions, rewards, danes, next_states = list(zip(*batch[k]))

                states = torch.FloatTensor(np.array(states))
                q_values, lstm_states = self._q(states, memories)

                next_states = torch.FloatTensor(np.array(next_states))
                next_q_values, next_lstm_states = self._q_target(next_states, lstm_states)
                memories = lstm_states

                for i in range(self.batch_size):
                    if danes[i]:
                        init_mem = self.get_initial_state(1)
                        memories[0][i] = init_mem[0]
                        memories[1][i] = init_mem[1]

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

    def reset(self):
        self._memory = self.get_initial_state(1)

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

    def _get_batch(self):
        batch_indexes = random.sample(range(len(self._buffer) - self.batch_len - 1), self.batch_size)
        batch = [list(map(lambda j: self._buffer[j + i], batch_indexes)) for i in range(self.batch_len)]
        return batch

    def _add_to_buffer(self, data):
        self._buffer.append(data)

        if len(self._buffer) > self.memory_size:
            self._buffer.pop(0)
