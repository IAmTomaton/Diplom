import copy
import random
import numpy as np
import torch


class DRQNAgent:

    def __init__(self, network, noise, state_dim, action_n, gamma=1, memory_size=30000, batch_size=32,
                 states_depth=4, burn_in=8, batch_len=12, learning_rate=1e-3, tau=1e-3, inf_mem_depth=False):
        """
        If inf_mem_depth is False, then we use an agent with finite memory depth, if True, then with infinite.
        """
        self._state_dim = state_dim
        self._action_n = action_n

        self.noise = noise

        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.states_depth = states_depth if not inf_mem_depth else 1
        self.burn_in = burn_in if inf_mem_depth else 0
        self.batch_len = batch_len
        self.learning_rate = learning_rate
        self.tau = tau
        self.inf_mem_depth = inf_mem_depth

        self._buffer = []
        self._q = network
        self._q_target = copy.deepcopy(self._q)
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

        self._memory = self.get_initial_state(1)
        self._prev_states = []

    def get_action(self, state, train=False):
        if not self._prev_states:
            self._prev_states = [state] * self.states_depth
        self._prev_states.pop(0)
        self._prev_states.append(state)

        if not self.inf_mem_depth:
            self._memory = self.get_initial_state(1)

        for state in self._prev_states:
            state = torch.FloatTensor(np.array([state]))

            if train:
                readouts, self._memory = self._q.step(state, self._memory)
            else:
                readouts, self._memory = self._q_target.step(state, self._memory)

        if train and np.random.uniform(0, 1) < self.noise.threshold:
            return self.noise.get()

        argmax_action = torch.argmax(readouts)
        return int(argmax_action)

    def fit_agent(self, state, action, reward, done, next_state):
        self._add_to_buffer([state, action, reward, done, next_state])

        if len(self._buffer) - self.states_depth > self.batch_size * self.batch_len:
            batch = self._get_batch()
            memories = self.get_initial_state(self.batch_size)
            target_memories = self.get_initial_state(self.batch_size)
            loss = 0

            for k in range(len(batch)):
                if k == self.burn_in:
                    h, c = memories
                    h.detach()
                    c.detach()

                states, actions, rewards, danes, next_states = list(zip(*batch[k]))

                states = torch.FloatTensor(np.array(states))
                q_values, memories = self._q(states, memories)
                for i in range(self.batch_size):
                    if danes[i]:
                        init_mem = self.get_initial_state(1)
                        memories[0][i] = init_mem[0]
                        memories[1][i] = init_mem[1]

                next_states = torch.FloatTensor(np.array(next_states))
                for i in range(self.batch_size):
                    if danes[i]:
                        init_mem = self.get_initial_state(1)
                        target_memories[0][i] = init_mem[0]
                        target_memories[1][i] = init_mem[1]
                next_q_values, target_memories = self._q_target(next_states, target_memories)

                if k >= self.burn_in and (k == self.states_depth - 1 or self.inf_mem_depth):
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
        self._prev_states = []
        self._memory = self.get_initial_state(1)

    def get_hyper_parameters(self):
        return {
            'agent_parameters': {
                'gamma': self.gamma,
                'memory_size': self.memory_size,
                'batch_size': self.batch_size,
                'inf_mem_depth': self.inf_mem_depth,
                'states_depth': self.states_depth,
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
        if self.inf_mem_depth:
            return [list(map(lambda j: self._buffer[j + i], batch_indexes)) for i in range(self.batch_len)]

        batch = [[self._buffer[j] for j in batch_indexes]]
        for i in range(1, self.states_depth):
            batch.append([self._buffer[j - i] if j - i >= 0 and not self._buffer[j - i][3] else batch[i - 1][index]
                          for index, j in enumerate(batch_indexes)])
        batch.reverse()
        return batch

    def _add_to_buffer(self, data):
        self._buffer.append(data)

        if len(self._buffer) > self.memory_size:
            self._buffer.pop(0)
