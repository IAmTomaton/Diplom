import copy
import math
import gym
import numpy as np
import torch
from torch import nn
from time import time
from log import save_log
from networks import *
from other.DubinsCar_Discrete import DubinsCar
from other.SimpleControlProblem_Discrete import SimpleControlProblem_Discrete
from train_info.epoch_log import EpochLog
from train_info.train_log import TrainLog
from utils import print_log


class DRQNROSTCDAgent(nn.Module):

    def __init__(self, network, state_dim, action_n, make_env, hyper_parameters, device):
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

        self._q = network
        self._q_target = copy.deepcopy(self._q)
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

        self._envs = [make_env() for _ in range(self.batch_size)]
        self._env_prev_states = [[env.reset()] * self.states_count for env in self._envs]
        self._rewards = [0 for _ in range(self.batch_size)]
        self._ended_rewards = []
        self._memory = []
        for _ in range(self.states_count):
            self.make_step()

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

    def fit_agent(self):
        batch = self._memory
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
        self._optimizer.step()
        self._optimizer.zero_grad()

        for target_param, param in zip(self._q_target.parameters(), self._q.parameters()):
            target_param.data.copy_((1 - self.st_coef) * target_param.data + self.st_coef * param.data)

        self.epsilon = max(self.min_epsilon, self.epsilon * self.mul_epsilon)

    def get_initial_state(self, batch_size):
        return self._q.get_initial_state(batch_size)

    def make_step(self):
        self._ended_rewards = []
        result = [self._make_step_in_env(i) for i in range(self.batch_size)]

        self._memory.append(result)
        if len(self._memory) > self.states_count:
            self._memory.pop(0)

        return self._ended_rewards

    def _make_step_in_env(self, i):
        env = self._envs[i]
        states = self._env_prev_states[i]
        state = states[-1]

        action = self.get_action(states, train=True)
        next_state, reward, done, _ = env.step(action)
        self._rewards[i] += reward

        if done:
            next_state = env.reset()
            self._ended_rewards.append(self._rewards[i])
            self._rewards[i] = 0

        states.pop(0)
        states.append(next_state)

        return state, action, reward, done, next_state


def get_session(agent, env):
    state = env.reset()
    states = [state] * agent.states_count
    done = False

    total_reward = 0
    while not done:
        action = agent.get_action(states, train=False)
        next_state, reward, done, _ = env.step(action)
        states.pop(0)
        states.append(next_state)
        total_reward += reward

        if done:
            break

    return total_reward


def train(env, agent, log_folder='logs', name='DRQNROSTCD', epoch_n=200, fit_n=500, test_n=20):
    train_log = TrainLog(name, agent.hyper_parameters)

    for epoch in range(epoch_n):
        t = time()
        epoch_rewards = []
        for _ in range(fit_n):
            agent.fit_agent()
            rewards = agent.make_step()
            epoch_rewards += rewards

        mean_reward = np.mean(epoch_rewards)

        test_rewards = [get_session(agent, env) for _ in range(test_n)]
        test_mean_reward = np.mean(test_rewards)
        std_dev = math.sqrt(np.mean([(r - test_mean_reward) ** 2 for r in test_rewards]))

        epoch_info = EpochLog(time() - t, mean_reward, epoch_rewards, test_mean_reward, test_rewards)
        train_log.add_epoch(epoch_info)

        save_log(train_log, log_folder + '\\' + train_log.name)
        print_log(epoch, mean_reward, time() - t, agent.epsilon, test_mean_reward, std_dev)


def make_env():
    env = gym.make("CartPole-v2")
    # env = DubinsCar()
    # env = SimpleControlProblem_Discrete()
    return env


def main():
    use_cuda = torch.cuda.is_available() and False
    device = torch.device('cuda' if use_cuda else 'cpu')
    env = make_env()
    print('Used', device)

    hyper_parameters = {'memory_size': 30000, 'gamma': 0.99, 'batch_size': 32, 'learning_rate': 1e-4,
                        'min_epsilon': 1e-5, 'mul_epsilon': 0.9999, 'states_count': 4, 'st_coef': 1e-3}

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    network = NetworkD72LSTM64D64(state_dim, action_n)
    agent = DRQNROSTCDAgent(network, state_dim, action_n, make_env, hyper_parameters, device)

    train(env, agent, 'logs', 'DRQNROSTCD_D72LSTM64D64_1')


if __name__ == '__main__':
    main()
