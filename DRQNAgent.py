import copy
import math
import random
import numpy as np
import torch


class DRQNAgent:

    def __init__(self, state_dim, action_n, q_modal, noise, gamma=1, memory_size=30000, batch_size=32,
                 history_len: int = 4, burn_in=8, trajectory_len=12, learning_rate=1e-3, tau=1e-3):
        """
        If inf_mem_depth is False, then we use an agent with finite memory depth, if True, then with infinite.
        """
        self._state_dim = state_dim
        self._action_n = action_n

        self.noise = noise

        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.history_len = history_len
        self._inf_history_len = math.isinf(history_len)
        self.burn_in = burn_in
        self.trajectory_len = max(trajectory_len, 1)
        self.learning_rate = learning_rate
        self.tau = tau

        self._real_history_len = self.history_len if not self._inf_history_len else 1

        self._memory = []
        self.q_model = q_modal
        self.q_target_model = copy.deepcopy(self.q_model)
        self._optimizer = torch.optim.Adam(self.q_model.parameters(), lr=self.learning_rate)

        self._hiddens = []
        self.reset()

    def get_action(self, state):
        state = torch.FloatTensor(np.array([state]))

        q_values, hidden = self.q_model(state, self._hiddens[0])
        action = np.argmax(q_values.data.numpy())

        if self._inf_history_len:
            self._hiddens[0] = hidden
        else:
            for i in range(1, len(self._hiddens)):
                _, hidden = self.q_model(state, self._hiddens[i])
                self._hiddens[i - 1] = hidden

            self._hiddens[-1] = self.get_initial_state(1)

        if np.random.uniform(0, 1) < self.noise.threshold:
            return self.noise.get()
        return action

    def fit(self, state, action, reward, done, next_state):
        self._add_to_memory([state, action, reward, done, next_state])

        if len(self._memory) - self._real_history_len > self.batch_size * self.trajectory_len:
            batch, valid_step_counts, make_burn_in = self._get_batch()
            hiddens = self.get_initial_state(self.batch_size)
            target_hiddens = self.get_initial_state(self.batch_size)
            loss = 0

            for k in range(len(batch)):
                if k == self.burn_in:
                    for i in range(self.batch_size):
                        if make_burn_in[i]:
                            hiddens[0][i].detach()
                            hiddens[1][i].detach()

                states, actions, rewards, danes, next_states = list(zip(*batch[k]))

                states = torch.FloatTensor(np.array(states))
                q_values, hiddens = self.q_model(states, hiddens)

                next_states = torch.FloatTensor(np.array(next_states))
                next_q_values, target_hiddens = self.q_target_model(next_states, target_hiddens)

                targets = q_values.clone()
                if self._inf_history_len or (not self._inf_history_len and k == len(batch) - 1):
                    for i in range(self.batch_size):
                        if not self._inf_history_len or \
                                ((not make_burn_in[i] or k >= self.burn_in) and k < valid_step_counts[i]):
                            targets[i][actions[i]] = rewards[i] + self.gamma * (1 - danes[i]) * max(next_q_values[i])
                    loss += torch.mean((targets.detach() - q_values) ** 2)

            if type(loss) is not int:
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()

                for target_param, param in zip(self.q_target_model.parameters(), self.q_model.parameters()):
                    target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

    def get_initial_state(self, batch_size):
        return self.q_model.get_initial_state(batch_size)

    def reset(self):
        self._hiddens = [self.get_initial_state(1) for _ in range(self._real_history_len)]

    def get_hyper_parameters(self):
        return {
            'agent_parameters': {
                'gamma': self.gamma,
                'memory_size': self.memory_size,
                'batch_size': self.batch_size,
                'history_len': self.history_len,
                'burn_in': self.burn_in,
                'trajectory_len': self.trajectory_len,
                'learning_rate': self.learning_rate,
                'tau': self.tau
            },
            'noise_parameters': self.noise.get_hyper_parameters(),
            'network_parameters': self.q_model.get_hyper_parameters(),
        }

    def _get_batch(self):
        count = self.trajectory_len if self._inf_history_len else self._real_history_len
        batch_indexes = random.sample(range(count, len(self._memory) - count), self.batch_size)

        starts = []
        burn_in = []
        for i in batch_indexes:
            start = i
            for j in range(1, count):
                start = i - j + 1
                if self._memory[i - j][3]:
                    break
            starts.append(start)
            burn_in.append(not self._memory[start - 1][3])

        ends = []
        for i in range(self.batch_size):
            center = batch_indexes[i]
            start = starts[i]
            end = center
            for j in range(count - center + start):
                end = center + j
                if self._memory[center + j][3]:
                    break
            ends.append(end)
        valid_step_count = [ends[i] - starts[i] + 1 for i in range(self.batch_size)]

        batch = []
        for j in range(count):
            batch_slice = []
            for i in range(self.batch_size):
                start = starts[i]
                center = batch_indexes[i]

                if self._inf_history_len:
                    batch_slice.append(self._memory[start + j])
                else:
                    current_index = center - count + 1 + j
                    if current_index < start:
                        batch_slice.append(self._memory[start])
                    else:
                        batch_slice.append(self._memory[current_index])

            batch.append(batch_slice)

        return batch, valid_step_count, burn_in

    def _add_to_memory(self, data):
        self._memory.append(data)

        if len(self._memory) > self.memory_size:
            self._memory.pop(0)
