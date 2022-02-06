import numpy as np
import torch
from torch import nn
import random
from collections import deque
from time import sleep


class NetworkLSTM(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.rnn_hidden_size = 64
        self.rnn_input_size = 72
        self.hid = nn.Linear(input_dim, self.rnn_input_size)
        self.lstm = nn.LSTMCell(self.rnn_input_size, self.rnn_hidden_size)
        self.logits = nn.Linear(self.rnn_hidden_size, output_dim)
        self.relu = nn.ReLU()

    def forward(self, prev_state, obs_state):
        hidden = self.hid(obs_state)
        hidden = self.relu(hidden)

        h_new, c_new = self.lstm(hidden, prev_state)

        logits = self.logits(h_new)

        return (h_new, c_new), logits

    def get_initial_state(self, batch_size):
        return torch.zeros((batch_size, self.rnn_hidden_size)), torch.zeros((batch_size, self.rnn_hidden_size))

    def step(self, prev_state, obs_t):
        (h, c), l = self.forward(prev_state, obs_t)
        return (h.detach(), c.detach()), l.detach()


class DRQNAgent(nn.Module):

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
        self._q = NetworkLSTM(self._state_dim, self._action_n)
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

    def get_action(self, prev_memories, state, train=False):
        state = torch.FloatTensor(np.array(state))
        new_memories, readouts = self._q.step(prev_memories, state)
        argmax_action = torch.argmax(readouts)

        if not train:
            return new_memories, int(argmax_action)

        probs = np.ones(self._action_n) * self.epsilon / self._action_n
        probs[argmax_action] += 1 - self.epsilon
        actions = np.arange(self._action_n)
        return new_memories, np.random.choice(actions, p=probs)

    def fit_DRQN(self, state, action, reward, done, next_state):
        self._memory.append([state, action, reward, done, next_state])
        if len(self._memory) > self.memory_size:
            self._memory.popleft()

        if len(self._memory) > self.batch_size:
            batch = random.sample(self._memory, self.batch_size)

            states, actions, rewards, danes, next_states = list(zip(*batch))
            states = torch.FloatTensor(np.array(states))
            q_values = self._q(states)
            next_states = torch.FloatTensor(np.array(next_states))
            next_q_values = self._q(next_states)
            targets = q_values.clone()
            for i in range(self.batch_size):
                targets[i][actions[i]] = rewards[i] + self.gamma * (1 - danes[i]) * max(next_q_values[i])

            loss = torch.mean((targets.detach() - q_values) ** 2)

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            self.epsilon = max(self.min_epsilon, self.epsilon * self.mul_epsilon)

    def get_initial_state(self, batch_size):
        return self._q.get_initial_state(batch_size)


def get_session(agent, env, batch_size=1, train_agent=False):
    state = env.reset()
    prev_memories = agent.get_initial_state(batch_size)

    total_reward = 0
    for _ in range(1000):
        new_memories, action = agent.get_action(prev_memories, [state], train=train_agent)
        next_state, reward, done, _ = env.step(action)
        if train_agent:
            agent.fit_DRQN(state, action, reward, done, next_state)
        state = next_state
        prev_memories = new_memories
        total_reward += reward

        if done:
            break

    return total_reward


def show_simulation(env, agent):
    state = env.reset()
    prev_memories = agent.get_initial_state(1)

    for t in range(1000):
        new_memories, action = agent.get_action(prev_memories, [state])
        next_state, reward, done, _ = env.step(action)
        env.render()
        sleep(0.02)
        state = next_state
        prev_memories = new_memories
        if done:
            break
    env.close()
