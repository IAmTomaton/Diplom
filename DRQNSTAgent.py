import copy
import math

import gym
import numpy as np
import torch
from torch import nn
from time import time
from train_info.train_log import TrainLog
from Buffer import Buffer
from train_info.epoch_log import EpochLog
from log import save_log
from networks import *
from other.SimpleControlProblem_Discrete import SimpleControlProblem_Discrete
from utils import print_log


class DRQNSTAgent(nn.Module):

    def __init__(self, network, state_dim, action_n, hyper_parameters, device):
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
        self.burn_in = hyper_parameters['burn_in']
        self.batch_len = hyper_parameters['batch_len']
        self.learning_rate = hyper_parameters['learning_rate']
        self.st_coef = hyper_parameters['st_coef']
        self.hyper_parameters = hyper_parameters

        self._memory = Buffer(self.memory_size)
        self._q = network
        self._q_target = copy.deepcopy(self._q)
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

    def get_action(self, prev_memories, state, train=False):
        state = torch.FloatTensor(np.array(state))

        if train:
            new_memories, readouts = self._q.step(prev_memories, state)
        else:
            new_memories, readouts = self._q_target.step(prev_memories, state)
        argmax_action = torch.argmax(readouts)

        if not train:
            return new_memories, int(argmax_action)

        probs = np.ones(self._action_n) * self.epsilon / self._action_n
        probs[argmax_action] += 1 - self.epsilon
        actions = np.arange(self._action_n)
        return new_memories, np.random.choice(actions, p=probs)

    def fit_agent(self, state, action, reward, done, next_state):
        self._memory.add([state, action, reward, done, next_state])

        if len(self._memory) > self.batch_size * (self.batch_len - self.burn_in):
            batch = self._memory.get_batch(self.batch_size, self.batch_len)
            memories = self.get_initial_state(self.batch_size)
            loss = 0

            for k in range(self.batch_len):
                states, actions, rewards, danes, next_states = list(zip(*batch[k]))

                states = torch.FloatTensor(np.array(states))
                lstm_states, q_values = self._q(memories, states)

                next_states = torch.FloatTensor(np.array(next_states))
                next_lstm_states, next_q_values = self._q_target(lstm_states, next_states)
                memories = lstm_states

                m = torch.zeros((self.batch_size, self.batch_size), device=self._device)
                for i in range(self.batch_size):
                    if not danes[i]:
                        m[i][i] = 1
                memories = torch.mm(m, memories[0]), torch.mm(m, memories[1])

                if k == self.burn_in:
                    h, c = memories
                    h.detach()
                    c.detach()

                if k >= self.burn_in:
                    targets = q_values.clone()
                    for i in range(self.batch_size):
                        targets[i][actions[i]] = rewards[i] + self.gamma * (1 - danes[i]) * max(next_q_values[i])

                    loss += torch.mean((targets.detach() - q_values) ** 2)

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            for target_param, param in zip(self._q_target.parameters(), self._q.parameters()):
                target_param.data.copy_((1 - self.st_coef) * target_param.data + self.st_coef * param.data)

            self.epsilon = max(self.min_epsilon, self.epsilon * self.mul_epsilon)

    def get_initial_state(self, batch_size):
        return self._q.get_initial_state(batch_size)


def get_session(agent, env, batch_size=1, train_agent=False):
    state = env.reset()
    prev_memories = agent.get_initial_state(batch_size)
    done = False

    total_reward = 0
    while not done:
        new_memories, action = agent.get_action(prev_memories, [state], train=train_agent)
        next_state, reward, done, _ = env.step(action)
        if train_agent:
            agent.fit_agent(state, action, reward, done, next_state)
        state = next_state
        prev_memories = new_memories
        total_reward += reward

        if done:
            break

    return total_reward


def train(env, agent, log_folder='logs', name='DRQNST', epoch_n=100, session_n=20, test_n=20):
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

        save_log(train_info, log_folder + '\\' + train_info.name)
        print_log(epoch, mean_reward, time() - t, agent.epsilon, test_mean_reward, std_dev)


def main():
    use_cuda = torch.cuda.is_available() and False
    device = torch.device('cuda' if use_cuda else 'cpu')
    env = gym.make("CartPole-v3")
    # env = DubinsCar()
    # env = SimpleControlProblem_Discrete()
    print('Used', device)

    hyper_parameters = {'memory_size': 30000, 'gamma': 0.99, 'batch_size': 32, 'learning_rate': 1e-4,
                        'min_epsilon': 1e-4, 'mul_epsilon': 0.9999, 'burn_in': 8, 'batch_len': 12, 'st_coef': 1e-3}

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    network = NetworkD72LSTM64D64(state_dim, action_n)
    agent = DRQNSTAgent(network, state_dim, action_n, hyper_parameters, device)

    train(env, agent, 'logs', 'DRQNST_D72LSTM64D64_4_v3')


if __name__ == '__main__':
    main()
