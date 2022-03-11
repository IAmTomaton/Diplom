import copy
import math
import gym
import numpy as np
import torch
from torch import nn
from time import time

from Noise import DiscreteUniformNoise
from SequentialNetwork import SequentialNetwork, LayerType
from log import save_log
from other.DubinsCar_Discrete import DubinsCar
from other.SimpleControlProblem_Discrete import SimpleControlProblem_Discrete
from train_info.epoch_log import EpochLog
from train_info.train_log import TrainLog
from utils import print_log


class DRQNROSTCDAgent(nn.Module):

    def __init__(self, network, noise, state_dim, action_n, gamma=1, batch_size=32, states_count=2,
                 learning_rate=1e-3, tau=1e-3):
        super().__init__()
        self._state_dim = state_dim
        self._action_n = action_n

        self.noise = noise

        self.gamma = gamma
        self.batch_size = batch_size
        self.states_count = states_count
        self.learning_rate = learning_rate
        self.tau = tau

        self._q = network
        self._q_target = copy.deepcopy(self._q)
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

    def get_action(self, states, train=False):
        if train and np.random.uniform(0, 1) < self.noise.threshold:
            return self.noise.get()

        memories = self.get_initial_state(1)

        for state in states:
            state_tensor = torch.FloatTensor(np.array([state]))

            if train:
                readouts, memories = self._q.step(state_tensor, memories)
            else:
                readouts, memories = self._q_target.step(state_tensor, memories)

        argmax_action = torch.argmax(readouts)
        return int(argmax_action)

    def fit_agent(self, batch):
        memories = self.get_initial_state(self.batch_size)
        next_memories = self.get_initial_state(self.batch_size)

        for k in range(self.states_count):
            states, actions, rewards, danes, next_states = list(zip(*batch[k]))

            states_tensor = torch.FloatTensor(np.array(states))
            q_values, memories = self._q(states_tensor, memories)

            next_states_tensor = torch.FloatTensor(np.array(next_states))
            next_q_values, next_memories = self._q_target(next_states_tensor, next_memories)

        targets = q_values.clone()
        for i in range(self.batch_size):
            targets[i][actions[i]] = rewards[i] + self.gamma * (1 - danes[i]) * max(next_q_values[i])

        loss = torch.mean((targets.detach() - q_values) ** 2)

        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

        for target_param, param in zip(self._q_target.parameters(), self._q.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

    def get_initial_state(self, batch_size):
        return self._q.get_initial_state(batch_size)

    def get_hyper_parameters(self):
        return {
            'agent_parameters': {
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'states_count': self.states_count,
                'learning_rate': self.learning_rate,
                'tau': self.tau
            },
            'noise_parameters': self.noise.get_hyper_parameters(),
            'network_parameters': self._q.get_hyper_parameters(),
        }


class Pool:

    def __init__(self, agent, make_env):
        self._agent = agent
        self._batch = []
        self._envs = [make_env() for _ in range(self._agent.batch_size)]
        self._env_prev_states = [[env.reset()] * self._agent.states_count for env in self._envs]
        self._rewards = [0 for _ in range(self._agent.batch_size)]
        self._ended_rewards = []
        for _ in range(self._agent.states_count):
            self.make_step()

    def make_step(self):
        self._ended_rewards = []
        result = [self._make_step_in_env(i) for i in range(self._agent.batch_size)]

        self._batch.append(result)
        if len(self._batch) > self._agent.states_count:
            self._batch.pop(0)

        return self._batch, self._ended_rewards

    def _make_step_in_env(self, i):
        env = self._envs[i]
        states = self._env_prev_states[i]
        state = states[-1]

        action = self._agent.get_action(states, train=True)
        next_state, reward, done, _ = env.step(action)
        self._rewards[i] += reward

        if done:
            next_state = env.reset()
            self._env_prev_states[i] = [next_state] * self._agent.states_count
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


def train(make_env, agent, log_folder='logs', name='DRQNROSTCD', epoch_n=200, fit_n=500, test_n=20):
    train_log = TrainLog(name, agent.get_hyper_parameters())
    env = make_env()
    pool = Pool(agent, make_env)

    for epoch in range(epoch_n):
        t = time()
        epoch_rewards = []
        for _ in range(fit_n):
            batch, rewards = pool.make_step()
            agent.fit_agent(batch)
            epoch_rewards += rewards
        agent.noise.reduce()

        mean_reward = np.mean(epoch_rewards)

        test_rewards = [get_session(agent, env) for _ in range(test_n)]
        test_mean_reward = np.mean(test_rewards)
        std_dev = math.sqrt(np.mean([(r - test_mean_reward) ** 2 for r in test_rewards]))

        epoch_info = EpochLog(time() - t, mean_reward, epoch_rewards, test_mean_reward, test_rewards)
        train_log.add_epoch(epoch_info)

        save_log(train_log, log_folder + '\\' + train_log.name)
        print_log(epoch, mean_reward, time() - t, test_mean_reward, std_dev)


def make_env():
    env = gym.make("CartPole-v1")
    # env = DubinsCar()
    # env = SimpleControlProblem_Discrete()
    return env


def main():
    env = make_env()

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    noise = DiscreteUniformNoise(action_n)
    network = SequentialNetwork(state_dim,
                                [(LayerType.Dense, 128),
                                 (LayerType.LSTM, 64),
                                 (LayerType.Dense, 32),
                                 (LayerType.Dense, action_n)],
                                nn.ReLU())
    agent = DRQNROSTCDAgent(network, noise, state_dim, action_n)

    train(make_env, agent, 'logs\\CartPole', 'DRQNROSTCD')


if __name__ == '__main__':
    main()
