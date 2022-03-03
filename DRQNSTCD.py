import copy
import math
import random
import gym
import numpy as np
import torch
from torch import nn
from time import time
from TrainLog import TrainLog
from EpochLog import EpochLog
from log import save_log
from other.DubinsCar_Discrete import DubinsCar
from other.SimpleControlProblem_Discrete import SimpleControlProblem_Discrete
from utils import print_log


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


class DRQNSTCDAgent(nn.Module):

    def __init__(self, state_dim, action_n, hyper_parameters, device):
        super().__init__()
        self._state_dim = state_dim
        self._action_n = action_n
        self._device = device

        self.epsilon = 1
        self.gamma = hyper_parameters['gamma']
        self.min_epsilon = hyper_parameters['min_epsilon']
        self.mul_epsilon = hyper_parameters['mul_epsilon']
        self.memory_size = hyper_parameters['memory_size']
        self.batch_size = hyper_parameters['batch_size']
        self.states_count = hyper_parameters['states_count']
        self.learning_rate = hyper_parameters['learning_rate']
        self.st_coef = hyper_parameters['st_coef']
        self.hyper_parameters = hyper_parameters

        self._memory = []
        self._q = NetworkLSTM(self._state_dim, self._action_n)
        self._q_target = copy.deepcopy(self._q)
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

    def get_action(self, states, train=False):
        memories = self.get_initial_state(1)

        for state in states:
            state_tensor = torch.FloatTensor(np.array([state]))

            if train:
                memories, readouts = self._q.step(memories, state_tensor)
            else:
                memories, readouts = self._q_target.step(memories, state_tensor)

        argmax_action = torch.argmax(readouts)

        if not train:
            return int(argmax_action)

        probs = np.ones(self._action_n) * self.epsilon / self._action_n
        probs[argmax_action] += 1 - self.epsilon
        actions = np.arange(self._action_n)
        return np.random.choice(actions, p=probs)

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
                memories, q_values = self._q(memories, states_tensor)

                next_states_tensor = torch.FloatTensor(np.array(next_states))
                next_memories, next_q_values = self._q_target(next_memories, next_states_tensor)

            targets = q_values.clone()
            for i in range(self.batch_size):
                targets[i][actions[i]] = rewards[i] + self.gamma * (1 - danes[i]) * max(next_q_values[i])

            loss = torch.mean((targets.detach() - q_values) ** 2)

            loss.backward()
            self._optimizer.zero_grad()
            self._optimizer.step()

            for target_param, param in zip(self._q_target.parameters(), self._q.parameters()):
                target_param.data.copy_((1 - self.st_coef) * target_param.data + self.st_coef * param.data)

            self.epsilon = max(self.min_epsilon, self.epsilon * self.mul_epsilon)

    def get_initial_state(self, batch_size):
        return self._q.get_initial_state(batch_size)

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

    total_reward = 0
    for _ in range(1000):
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


def train(env, agent, log_folder='logs', name='DRQNSTCD', epoch_n=200, session_n=20, test_n=100):
    train_info = TrainLog(name, agent.hyper_parameters)

    for epoch in range(epoch_n):
        t = time()
        rewards = [get_session(agent, env, train_agent=True) for _ in range(session_n)]
        mean_reward = np.mean(rewards)

        test_rewards = [get_session(agent, env) for _ in range(test_n)]
        test_mean_reward = np.mean(test_rewards)
        std_dev = math.sqrt(np.mean([(r - test_mean_reward) ** 2 for r in test_rewards]))

        epoch_info = EpochLog(time() - t, mean_reward, rewards, test_mean_reward, test_rewards)
        train_info.add_epoch(epoch_info)

        save_log(train_info, log_folder + '\\' + train_info.name + '_log' + '.json')
        print_log(epoch, mean_reward, time() - t, agent.epsilon, test_mean_reward, std_dev)


def main():
    use_cuda = torch.cuda.is_available() and False
    device = torch.device('cuda' if use_cuda else 'cpu')
    # env = gym.make("CartPole-v1")
    # env = DubinsCar()
    env = SimpleControlProblem_Discrete()
    print('Used', device)

    hyper_parameters = {'memory_size': 30000, 'gamma': 0.95, 'batch_size': 32, 'learning_rate': 1e-4,
                        'min_epsilon': 1e-4, 'mul_epsilon': 0.9999, 'states_count': 3, 'st_coef': 1e-3}

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    agent = DRQNSTCDAgent(state_dim, action_n, hyper_parameters, device)

    train(env, agent, 'logs_SimpleControlProblem_Discrete')


if __name__ == '__main__':
    main()
