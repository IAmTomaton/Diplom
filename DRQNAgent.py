import copy
import math
import random
import numpy as np
import torch


def _detach_hiddens(hiddens, starts_with_sessions):
    hidden_list = list(torch.split(hiddens[0], 1)), list(torch.split(hiddens[1], 1))
    for i in range(len(starts_with_sessions)):
        if not starts_with_sessions[i]:
            hidden_list[0][i] = hidden_list[0][i].detach()
            hidden_list[1][i] = hidden_list[1][i].detach()
    hiddens = torch.cat(hidden_list[0]), torch.cat(hidden_list[1])
    return hiddens


class DRQNAgent:

    def __init__(self, state_dim, action_n, q_modal, noise, gamma=1, memory_size=30000, batch_size=32,
                 history_len=4, burn_in=8, trajectory_len=12, learning_rate=1e-3, tau=1e-3, name='DRQN'):
        self._state_dim = state_dim
        self._action_n = action_n

        self.name = name

        self.noise = noise

        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.history_len = history_len
        self._inf_history_len = history_len == 'inf' or math.isinf(history_len)
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
            trajectories, starts_with_sessions = self._get_batch_trajectories()
            hiddens = self.get_initial_state(self.batch_size)
            target_hiddens = self.get_initial_state(self.batch_size)
            loss = 0

            for k in range(len(max(trajectories, key=len))):
                if k == self.burn_in and self._inf_history_len:
                    # Детачим hiddens для тех траекторий для которых нужен burn_in
                    hiddens = _detach_hiddens(hiddens, starts_with_sessions)

                slice_trajectories = [trajectory[k] for trajectory in trajectories]
                states, actions, rewards, danes, next_states = list(zip(*slice_trajectories))

                states = torch.FloatTensor(np.array(states))
                q_values, hiddens = self.q_model(states, hiddens)

                next_states = torch.FloatTensor(np.array(next_states))
                next_q_values, target_hiddens = self.q_target_model(next_states, target_hiddens)

                if not self._inf_history_len and k == self.history_len - 1:
                    loss += self._calculate_loss(q_values, actions, rewards, danes, next_q_values)

                if self._inf_history_len:
                    if k >= self.burn_in:
                        loss += self._calculate_loss(q_values, actions, rewards, danes, next_q_values)
                    elif any(starts_with_sessions):
                        # Считаем loss для траекторий начало которых совпадает с началом сессии,
                        # а значит для них не нужен burn_in и мы можем сразу считать loss
                        indices = [i for i, x in enumerate(starts_with_sessions) if x]
                        q_values = torch.index_select(q_values, 0, torch.tensor(indices))
                        next_q_values = torch.index_select(next_q_values, 0, torch.tensor(indices))
                        loss += self._calculate_loss(q_values, actions, rewards, danes, next_q_values)

                    indices = [i for i in range(len(trajectories)) if len(trajectories[i]) - 1 > k]
                    if not indices:
                        break
                    if len(indices) != len(trajectories):
                        # Удаляем закончившееся траектории и всё что с ними связанно
                        trajectories = [trajectories[i] for i in indices]
                        starts_with_sessions = [starts_with_sessions[i] for i in indices]
                        indices = torch.tensor(indices)
                        hiddens = torch.index_select(hiddens[0], 0, indices), torch.index_select(hiddens[1], 0, indices)
                        target_hiddens = torch.index_select(target_hiddens[0], 0, indices), \
                                         torch.index_select(target_hiddens[1], 0, indices)

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
                'name': self.name,
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

    def _calculate_loss(self, q_values, actions, rewards, danes, next_q_values):
        targets = q_values.clone()
        for i in range(q_values.size(0)):
            targets[i][actions[i]] = rewards[i] + self.gamma * (1 - danes[i]) * max(next_q_values[i])
        return torch.mean((targets.detach() - q_values) ** 2)

    def _get_batch_trajectories(self):
        count = self.trajectory_len if self._inf_history_len else self._real_history_len
        centers = random.sample(range(count, len(self._memory) - count), self.batch_size)

        # Находим начало траекторий
        starts = []
        starts_with_sessions = []
        for center in centers:
            start = self._find_trajectory_start(center, count)
            starts.append(start)
            starts_with_sessions.append(self._memory[start - 1][3])

        # Находим конец траекторий
        ends = centers
        if self._inf_history_len:
            ends = [self._find_trajectory_end(centers[i], starts[i], count) for i in range(self.batch_size)]

        # Собираем траектории
        batch = []
        for i in range(self.batch_size):
            start = starts[i]
            end = ends[i]
            if self._inf_history_len:
                trajectory = self._memory[start: end + 1]
            else:
                real_trajectory_len = end - start + 1
                repeat_start_count = count - real_trajectory_len
                trajectory = [self._memory[start]] * repeat_start_count + self._memory[start: end + 1]

            batch.append(trajectory)

        return batch, starts_with_sessions

    def _find_trajectory_start(self, center, count):
        start = center
        for j in range(1, count):
            start = center - j + 1
            if self._memory[start - 1][3]:
                break
        return start

    def _find_trajectory_end(self, center, start, count):
        end = center
        for j in range(count - center + start):
            end = center + j
            if self._memory[end][3]:
                break
        return end

    def _add_to_memory(self, data):
        self._memory.append(data)

        if len(self._memory) > self.memory_size:
            self._memory.pop(0)
