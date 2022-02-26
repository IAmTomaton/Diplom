import copy
import math
import gym
import numpy as np
import torch
from torch import nn
from time import time
from EpochLog import EpochLog
from TrainLog import TrainLog
from log import save_log
from other.DubinsCar_Discrete import DubinsCar
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


class DRQNROSTAgent(nn.Module):

    def __init__(self, state_dim, action_n, make_env, hyper_parameters, device):
        super().__init__()
        self._state_dim = state_dim
        self._action_n = action_n
        self._device = device

        self.gamma = hyper_parameters['gamma']
        self.epsilon = 1
        self.min_epsilon = hyper_parameters['min_epsilon']
        self.mul_epsilon = hyper_parameters['mul_epsilon']
        self.batch_size = hyper_parameters['batch_size']
        self.episode_len = hyper_parameters['episode_len']
        self.learning_rate = hyper_parameters['learning_rate']
        self.st_coef = hyper_parameters['st_coef']
        self.hyper_parameters = hyper_parameters

        self._q = NetworkLSTM(self._state_dim, self._action_n)
        self._q_target = copy.deepcopy(self._q)
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

        self._envs = [make_env() for _ in range(self.batch_size)]
        self._memories = [self.get_initial_state(1) for _ in range(self.batch_size)]
        self._env_states = [env.reset() for env in self._envs]
        self._rewards = [0 for _ in range(self.batch_size)]
        self._ended_rewards = []

    def get_action(self, memory, state, train=False):
        state = torch.FloatTensor(np.array(state))

        if train:
            new_memory, readouts = self._q.step(memory, state)
        else:
            new_memory, readouts = self._q_target.step(memory, state)
        argmax_action = torch.argmax(readouts)

        if not train:
            return new_memory, int(argmax_action)

        probs = np.ones(self._action_n) * self.epsilon / self._action_n
        probs[argmax_action] += 1 - self.epsilon
        actions = np.arange(self._action_n)
        return new_memory, np.random.choice(actions, p=probs)

    def make_step(self, i):
        memory = self._memories[i]
        state = self._env_states[i]
        env = self._envs[i]

        new_memory, action = self.get_action(memory, [state], train=True)
        next_state, reward, done, _ = env.step(action)
        self._rewards[i] += reward

        if done:
            new_memory = self.get_initial_state(1)
            next_state = env.reset()
            self._ended_rewards.append(self._rewards[i])
            self._rewards[i] = 0

        self._memories[i] = new_memory
        self._env_states[i] = next_state

        return state, action, reward, done, next_state

    def get_batch(self):
        self._ended_rewards = []
        for _ in range(self.episode_len):
            yield [self.make_step(i) for i in range(self.batch_size)]

    def fit_agent(self):
        memories = [mem for mem in self._memories]
        for h, c in memories:
            h.detach()
            c.detach()

        loss = 0
        batch = list(self.get_batch())

        for SADSes in batch:
            for i in range(self.batch_size):
                state, action, reward, done, next_state = SADSes[i]
                memory = memories[i]

                state = torch.FloatTensor(np.array([state]))
                lstm_state, q_value = self._q(memory, state)

                next_state = torch.FloatTensor(np.array([next_state]))
                next_lstm_state, next_q_value = self._q_target(lstm_state, next_state)

                if done:
                    lstm_state = self.get_initial_state(1)
                memories[i] = lstm_state

                target = q_value.clone()
                target[0][action] = reward + self.gamma * (1 - done) * max(next_q_value[0])
                loss += torch.mean((target.detach() - q_value) ** 2)

        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.mul_epsilon)

        for target_param, param in zip(self._q_target.parameters(), self._q.parameters()):
            target_param.data.copy_((1 - self.st_coef) * target_param.data + self.st_coef * param.data)

        return self._ended_rewards

    def get_initial_state(self, batch_size):
        return self._q.get_initial_state(batch_size)


def get_session(agent, env):
    state = env.reset()
    prev_memories = agent.get_initial_state(1)

    total_reward = 0
    for _ in range(1000):
        new_memories, action = agent.get_action(prev_memories, [state], train=False)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        prev_memories = new_memories
        total_reward += reward

        if done:
            break

    return total_reward


def train(env, agent, log_folder='logs', name='DRQNROST', epoch_n=1000, episode_n=200, test_n=100):
    train_info = TrainLog(name, agent.hyper_parameters)

    for epoch in range(epoch_n):
        t = time()
        epoch_rewards = []
        for _ in range(episode_n):
            rewards = agent.fit_agent()
            epoch_rewards += rewards

        mean_reward = np.mean(epoch_rewards)
        test_rewards = [get_session(agent, env) for _ in range(test_n)]
        test_mean_reward = np.mean(test_rewards)

        std_dev = math.sqrt(np.mean([(r - test_mean_reward) ** 2 for r in test_rewards]))

        epoch_info = EpochLog(time() - t, mean_reward, epoch_rewards, test_mean_reward, test_rewards)
        train_info.add_epoch(epoch_info)

        save_log(train_info, log_folder + '\\' + train_info.name + '_log' + '.json')
        print_log(epoch, mean_reward, time() - t, agent.epsilon, test_mean_reward, std_dev)


def make_env():
    # env = gym.make("CartPole-v1")
    env = DubinsCar()
    return env


def main():
    use_cuda = torch.cuda.is_available() and False
    device = torch.device('cuda' if use_cuda else 'cpu')
    env = make_env()
    print('Used', device)

    hyper_parameters = {'gamma': 0.95, 'batch_size': 16, 'learning_rate': 1e-4, 'min_epsilon': 1e-4,
                        'mul_epsilon': 0.9999, 'episode_len': 4, 'st_coef': 1e-3}

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    agent = DRQNROSTAgent(state_dim, action_n, make_env, hyper_parameters, device)

    train(env, agent, 'logs_DubinsCar')


if __name__ == '__main__':
    main()
