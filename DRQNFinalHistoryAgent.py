import copy
import random
import numpy as np
import torch


class DRQNFinalHistoryAgent:

    def __init__(self, state_dim, action_n, q_modal, noise, gamma=1, memory_size=30000, batch_size=32,
                 history_len=4, learning_rate=1e-3, tau=1e-3, name='DRQNFinalHistory'):
        self._state_dim = state_dim
        self._action_n = action_n

        self.name = name

        self.noise = noise

        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.history_len = history_len
        self.learning_rate = learning_rate
        self.tau = tau

        self._memory = []
        self.q_model = q_modal
        self.q_target_model = copy.deepcopy(self.q_model)
        self._optimizer = torch.optim.Adam(self.q_model.parameters(), lr=self.learning_rate)

        self._hiddens_stack = []
        self.reset()

    def get_action(self, state):
        state = torch.FloatTensor(np.array([state]))

        q_values, hidden = self.q_model(state, self._hiddens_stack[0])
        action = np.argmax(q_values.data.numpy())

        for i in range(1, len(self._hiddens_stack)):
            _, hidden = self.q_model(state, self._hiddens_stack[i])
            self._hiddens_stack[i - 1] = hidden

        self._hiddens_stack[-1] = self.get_initial_hiddens(1)

        if np.random.uniform(0, 1) < self.noise.threshold:
            return self.noise.get()
        return action

    def fit(self, state, action, reward, done, next_state):
        self._add_to_memory([state, action, reward, done, next_state])

        if len(self._memory) > self.batch_size * self.history_len:
            trajectories, starts_with_sessions = self._get_batch_trajectories()
            hiddens = self.get_initial_hiddens(self.batch_size)
            target_hiddens = self.get_initial_hiddens(self.batch_size)

            for k in range(self.history_len):
                slice_trajectories = [trajectory[k] for trajectory in trajectories]
                states, actions, rewards, danes, next_states = list(zip(*slice_trajectories))

                states = torch.FloatTensor(np.array(states))
                q_values, hiddens = self.q_model(states, hiddens)

                next_states = torch.FloatTensor(np.array(next_states))
                next_q_values, target_hiddens = self.q_target_model(next_states, target_hiddens)

            loss = self._calculate_loss(q_values, actions, rewards, danes, next_q_values)

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            for target_param, param in zip(self.q_target_model.parameters(), self.q_model.parameters()):
                target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

    def get_initial_hiddens(self, batch_size):
        return self.q_model.get_initial_state(batch_size)

    def reset(self):
        self._hiddens_stack = [self.get_initial_hiddens(1) for _ in range(self.history_len)]

    def get_hyper_parameters(self):
        return {
            'agent_parameters': {
                'name': self.name,
                'gamma': self.gamma,
                'memory_size': self.memory_size,
                'batch_size': self.batch_size,
                'history_len': self.history_len,
                'learning_rate': self.learning_rate,
                'tau': self.tau
            },
            'noise_parameters': self.noise.get_hyper_parameters(),
            'network_parameters': self.q_model.get_hyper_parameters(),
        }

    def _calculate_loss(self, q_values, actions, rewards, danes, next_q_values):
        targets = q_values.clone()
        for i in range(q_values.size(0)):
            targets[i][actions[i]] = rewards[i] + self.gamma * (1 - danes[i]) * max(next_q_values[i])
        return torch.mean((targets.detach() - q_values) ** 2)

    def _get_batch_trajectories(self):
        count = self.history_len
        ends = random.sample(range(count, len(self._memory) - count), self.batch_size)

        # Находим начало траекторий
        starts = []
        starts_with_sessions = []
        for center in ends:
            start = self._find_trajectory_start(center, count)
            starts.append(start)
            starts_with_sessions.append(self._memory[start - 1][3])

        # Собираем траектории
        batch = []
        for i in range(self.batch_size):
            trajectory = self._memory[starts[i]: ends[i] + 1]
            trajectory = [self._memory[starts[i]]] * (count - len(trajectory)) + trajectory
            batch.append(trajectory)

        return batch, starts_with_sessions

    def _find_trajectory_start(self, center, count):
        for j in range(1, count):
            start = center - j + 1
            if self._memory[start - 1][3]:
                return start
        return center - count + 1

    def _add_to_memory(self, data):
        self._memory.append(data)

        if len(self._memory) > self.memory_size:
            self._memory.pop(0)
