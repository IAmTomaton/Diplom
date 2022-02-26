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
from utils import print_log


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


class DQNROSTAgent(nn.Module):

    def __init__(self, state_dim, action_n, make_env, hyper_parameters, device):
        super().__init__()
        self._state_dim = state_dim
        self._action_n = action_n
        self._device = device

        self.epsilon = 1
        self.gamma = hyper_parameters['gamma']
        self.min_epsilon = hyper_parameters['min_epsilon']
        self.mul_epsilon = hyper_parameters['mul_epsilon']
        self.batch_size = hyper_parameters['batch_size']
        self.episode_len = hyper_parameters['episode_len']
        self.learning_rate = hyper_parameters['learning_rate']
        self.st_coefficient = hyper_parameters['st_coefficient']
        self.hyper_parameters = hyper_parameters

        self._q = Network(self._state_dim, self._action_n)
        self._q_target = copy.deepcopy(self._q)
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

        self._envs = [make_env() for _ in range(self.batch_size)]
        self._env_states = [env.reset() for env in self._envs]
        self._rewards = [0 for _ in range(self.batch_size)]
        self._ended_rewards = []

    def get_action(self, state, train=False):
        state = torch.FloatTensor(np.array(state))

        if train:
            argmax_action = torch.argmax(self._q(state))
        else:
            argmax_action = torch.argmax(self._q_target(state))
            return int(argmax_action)

        probs = np.ones(self._action_n) * self.epsilon / self._action_n
        probs[argmax_action] += 1 - self.epsilon
        actions = np.arange(self._action_n)
        return np.random.choice(actions, p=probs)

    def make_step(self, i):
        state = self._env_states[i]
        env = self._envs[i]

        action = self.get_action([state], train=True)
        next_state, reward, done, _ = env.step(action)
        self._rewards[i] += reward

        if done:
            next_state = env.reset()
            self._ended_rewards.append(self._rewards[i])
            self._rewards[i] = 0

        self._env_states[i] = next_state

        return state, action, reward, done, next_state

    def get_batch(self):
        self._ended_rewards = []
        for _ in range(self.episode_len):
            yield [self.make_step(i) for i in range(self.batch_size)]

    def fit_agent(self):
        loss = 0
        batch = list(self.get_batch())

        for SADSes in batch:
            for i in range(self.batch_size):
                state, action, reward, done, next_state = SADSes[i]

                state = torch.FloatTensor(np.array(state))
                q_value = self._q(state)

                next_state = torch.FloatTensor(np.array(next_state))
                next_q_value = self._q_target(next_state)

                target = q_value.clone()
                target[action] = reward + self.gamma * (1 - done) * max(next_q_value)
                loss = torch.mean((target.detach() - q_value) ** 2)

        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.mul_epsilon)

        for target_param, param in zip(self._q_target.parameters(), self._q.parameters()):
            target_param.data.copy_((1 - self.st_coefficient) * target_param.data + self.st_coefficient * param.data)

        return self._ended_rewards


def get_session(agent, env, train_agent=False):
    state = env.reset()
    total_reward = 0
    for _ in range(1000):
        action = agent.get_action(state, train=train_agent)
        next_state, reward, done, _ = env.step(action)
        if train_agent:
            agent.fit_DQN(state, action, reward, done, next_state)
        state = next_state
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
    return gym.make("CartPole-v1")


def main():
    use_cuda = torch.cuda.is_available() and False
    device = torch.device('cuda' if use_cuda else 'cpu')
    env = make_env()
    print('Used', device)

    hyper_parameters = {'gamma': 0.95, 'batch_size': 64, 'learning_rate': 1e-4,
                        'min_epsilon': 1e-4, 'mul_epsilon': 0.9999, 'episode_len': 2, 'st_coefficient': 1e-3}

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    agent = DQNROSTAgent(state_dim, action_n, make_env, hyper_parameters, device)

    train(env, agent)


if __name__ == '__main__':
    main()
