import math
import numpy as np
import torch
from torch import nn
from time import time
from train_info.epoch_log import EpochLog
from train_info.train_log import TrainLog
from log import save_log
from networks.NetworkD64D64 import NetworkD64D64
from other.DubinsCar_Discrete import DubinsCar
from utils import print_log


class DQNROAgent(nn.Module):

    def __init__(self, network, state_dim, action_n, make_env, hyper_parameters, device):
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
        self.hyper_parameters = hyper_parameters

        self._q = network
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

        self._envs = [make_env() for _ in range(self.batch_size)]
        self._env_states = [env.reset() for env in self._envs]
        self._rewards = [0 for _ in range(self.batch_size)]
        self._ended_rewards = []

    def get_action(self, state, train=False):
        state = torch.FloatTensor(np.array(state))
        argmax_action = torch.argmax(self._q(state))

        if not train:
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

        for SARDSes in batch:
            for i in range(self.batch_size):
                state, action, reward, done, next_state = SARDSes[i]

                state = torch.FloatTensor(np.array(state))
                q_value = self._q(state)
                next_state = torch.FloatTensor(np.array(next_state))
                next_q_value = self._q(next_state)

                target = q_value.clone()
                target[action] = reward + self.gamma * (1 - done) * max(next_q_value)
                loss += torch.mean((target.detach() - q_value) ** 2)

        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.mul_epsilon)

        return self._ended_rewards, int(loss)


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


def train(env, agent, log_folder='logs', name='DQNRO', epoch_n=1000, episode_n=200, test_n=100):
    train_info = TrainLog(name, agent.hyper_parameters)

    for epoch in range(epoch_n):
        t = time()
        epoch_rewards = []
        epoch_loss = []
        for _ in range(episode_n):
            rewards, loss = agent.fit_agent()
            epoch_loss.append(loss)
            epoch_rewards += rewards

        print(np.mean(epoch_loss))

        mean_reward = np.mean(epoch_rewards)
        test_rewards = [get_session(agent, env) for _ in range(test_n)]
        test_mean_reward = np.mean(test_rewards)
        std_dev = math.sqrt(np.mean([(r - test_mean_reward) ** 2 for r in test_rewards]))

        epoch_info = EpochLog(time() - t, mean_reward, epoch_rewards, test_mean_reward, test_rewards)
        train_info.add_epoch(epoch_info)

        save_log(train_info, log_folder + '\\' + train_info.name)
        print_log(epoch, mean_reward, time() - t, agent.epsilon, test_mean_reward, std_dev)


def make_env():
    # env = gym.make("CartPole-v1")
    env = DubinsCar()
    # env = SimpleControlProblem_Discrete()
    return env


def main():
    use_cuda = torch.cuda.is_available() and False
    device = torch.device('cuda' if use_cuda else 'cpu')
    env = make_env()
    print('Used', device)

    hyper_parameters = {'gamma': 0.95, 'batch_size': 32, 'learning_rate': 1e-4,
                        'min_epsilon': 1e-4, 'mul_epsilon': 0.9999, 'episode_len': 2}

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    network = NetworkD64D64(state_dim, action_n)
    agent = DQNROAgent(network, state_dim, action_n, make_env, hyper_parameters, device)

    train(env, agent, 'logs_DubinsCar')


if __name__ == '__main__':
    main()
