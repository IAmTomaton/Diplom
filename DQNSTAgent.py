import copy
import math
import gym
import numpy as np
import torch
from torch import nn
import random
from collections import deque
from time import sleep, time
from EpochLog import EpochLog
from TrainLog import TrainLog
from log import save_log
from other.DubinsCar_Discrete import DubinsCar
from other.SimpleControlProblem_Discrete import SimpleControlProblem_Discrete
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


class DQNAgent(nn.Module):

    def __init__(self, state_dim, action_n, hyper_parameters, device):
        super().__init__()
        self._state_dim = state_dim
        self._action_n = action_n
        self._device = device

        self.gamma = hyper_parameters['gamma']
        self.epsilon = 1
        self.min_epsilon = hyper_parameters['min_epsilon']
        self.mul_epsilon = hyper_parameters['mul_epsilon']
        self.memory_size = hyper_parameters['memory_size']
        self.batch_size = hyper_parameters['batch_size']
        self.learning_rate = hyper_parameters['learning_rate']
        self.st_coef = hyper_parameters['st_coef']
        self.hyper_parameters = hyper_parameters

        self._memory = deque()
        self._q = Network(self._state_dim, self._action_n)
        self._q_target = copy.deepcopy(self._q)
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

    def get_action(self, state, train=False):
        state = torch.FloatTensor(np.array(state)).to(device=self._device, non_blocking=True)
        if not train:
            argmax_action = torch.argmax(self._q_target(state))
            return int(argmax_action)
        else:
            argmax_action = torch.argmax(self._q(state))

        probs = np.ones(self._action_n) * self.epsilon / self._action_n
        probs[argmax_action] += 1 - self.epsilon
        actions = np.arange(self._action_n)
        return np.random.choice(actions, p=probs)

    def fit_DQN(self, state, action, reward, done, next_state):
        self._memory.append([state, action, reward, done, next_state])

        if len(self._memory) > self.memory_size:
            self._memory.popleft()

        if len(self._memory) > self.batch_size:
            batch = random.sample(self._memory, self.batch_size)

            states, actions, rewards, danes, next_states = list(zip(*batch))
            states = torch.FloatTensor(np.array(states))
            q_values = self._q(states)
            next_states = torch.FloatTensor(np.array(next_states))
            next_q_values = self._q_target(next_states)
            targets = q_values.clone()
            for i in range(self.batch_size):
                targets[i][actions[i]] = rewards[i] + self.gamma * (1 - danes[i]) * max(next_q_values[i])

            loss = torch.mean((targets.detach() - q_values) ** 2)

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            for target_param, param in zip(self._q_target.parameters(), self._q.parameters()):
                target_param.data.copy_((1 - self.st_coef) * target_param.data + self.st_coef * param.data)

            self.epsilon = max(self.min_epsilon, self.epsilon * self.mul_epsilon)

            return loss.item()

        return 0


def get_session(agent, env, train_agent=False):
    state = env.reset()
    total_reward = 0
    loss_mean = 0
    for _ in range(1000):
        action = agent.get_action(state, train=train_agent)
        next_state, reward, done, _ = env.step(action)
        if train_agent:
            loss = agent.fit_DQN(state, action, reward, done, next_state)
            loss_mean = (loss_mean + loss) / 2
        state = next_state
        total_reward += reward

        if done:
            break

    return total_reward, loss_mean


def show_simulation(env, agent):
    state = env.reset()
    for t in range(1000):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        env.render()
        sleep(0.02)
        state = next_state
        if done:
            break
    env.close()


def train(env, agent, log_folder='logs', name='DQNST', epoch_n=100, session_n=100, test_n=100):
    train_info = TrainLog(name, agent.hyper_parameters)

    for epoch in range(epoch_n):
        t = time()
        rewards, loss = list(zip(*[get_session(agent, env, train_agent=True) for _ in range(session_n)]))
        print(np.mean(loss))
        mean_reward = np.mean(rewards)

        test_rewards, loss = list(zip(*[get_session(agent, env) for _ in range(test_n)]))
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
    # env = SimpleControlProblem_Discrete()
    env = DubinsCar()
    print('Used', device)

    hyper_parameters = {'memory_size': 30000, 'gamma': 0.95, 'batch_size': 64, 'learning_rate': 1e-4,
                        'min_epsilon': 1e-4, 'mul_epsilon': 0.9999, 'st_coef': 1e-3}

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    agent = DQNAgent(state_dim, action_n, hyper_parameters, device)

    train(env, agent, 'logs_DubinsCar')


if __name__ == '__main__':
    main()
