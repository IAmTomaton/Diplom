import copy
import math
import numpy as np
import torch
import gym
from torch import nn
from time import time

from SequentialNetwork import SequentialNetwork, LayerType
from other.Noises import DiscreteUniformNoise
from other.SimpleControlProblem_Discrete import SimpleControlProblem_Discrete
from train_info.epoch_log import EpochLog
from train_info.train_log import TrainLog
from log import save_log
from other.DubinsCar_Discrete import DubinsCar
from utils import print_log


class DRQNROSTAgent(nn.Module):

    def __init__(self, network, noise, state_dim, action_n, gamma=1, episode_n=2, batch_size=32,
                 learning_rate=1e-3, tau=1e-3):
        super().__init__()
        self._state_dim = state_dim
        self._action_n = action_n

        self.noise = noise

        self.gamma = gamma
        self.batch_size = batch_size
        self.episode_n = episode_n
        self.learning_rate = learning_rate
        self.tau = tau

        self._q = network
        self._q_target = copy.deepcopy(self._q)
        self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self.learning_rate)

    def get_action(self, state, prev_memories, train=False):
        state = torch.FloatTensor(np.array([state]))

        if train:
            readouts, new_memories = self._q.step(state, prev_memories)
            if np.random.uniform(0, 1) < self.noise.threshold:
                return self.noise.get(), new_memories
        else:
            readouts, new_memories = self._q_target.step(state, prev_memories)

        argmax_action = torch.argmax(readouts)
        return int(argmax_action), new_memories

    def fit_agent(self, batch, init_memories):
        loss = 0

        for batch_slice in batch:
            for i in range(self.batch_size):
                state, action, reward, done, next_state = batch_slice[i]
                memory = init_memories[i]

                state = torch.FloatTensor(np.array([state]))
                q_value, lstm_state = self._q(state, memory)

                next_state = torch.FloatTensor(np.array([next_state]))
                next_q_value, next_lstm_state = self._q_target(next_state, lstm_state)

                if done:
                    lstm_state = self.get_initial_state(1)
                init_memories[i] = lstm_state

                target = q_value.clone()
                target[0][action] = reward + self.gamma * (1 - done) * max(next_q_value[0])
                loss += torch.mean((target.detach() - q_value) ** 2)

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
                'episode_n': self.episode_n,
                'learning_rate': self.learning_rate,
                'tau': self.tau
            },
            'noise_parameters': self.noise.get_hyper_parameters(),
            'network_parameters': self._q.get_hyper_parameters(),
        }


class Pool:

    def __init__(self, agent, make_env):
        self._agent = agent
        self._envs = [make_env() for _ in range(self._agent.batch_size)]
        self._memories = [self._agent.get_initial_state(1) for _ in range(self._agent.batch_size)]
        self._env_states = [env.reset() for env in self._envs]
        self._rewards = [0 for _ in range(self._agent.batch_size)]
        self._ended_rewards = []

    def get_memories(self):
        memories = [mem for mem in self._memories]
        for h, c in memories:
            h.detach()
            c.detach()
        return memories

    def make_step(self):
        self._ended_rewards = []
        batch = []
        for _ in range(self._agent.episode_n):
            batch.append([self._make_step_in_env(i) for i in range(self._agent.batch_size)])
        return batch, self._ended_rewards

    def _make_step_in_env(self, i):
        memory = self._memories[i]
        state = self._env_states[i]
        env = self._envs[i]

        action, new_memory = self._agent.get_action(state, memory, train=True)
        next_state, reward, done, _ = env.step(action)
        self._rewards[i] += reward

        self._env_states[i] = next_state
        self._memories[i] = new_memory

        if done:
            self._memories[i] = self._agent.get_initial_state(1)
            self._env_states[i] = env.reset()
            self._ended_rewards.append(self._rewards[i])
            self._rewards[i] = 0

        return state, action, reward, done, next_state


def get_session(agent, env):
    state = env.reset()
    prev_memories = agent.get_initial_state(1)
    done = False

    total_reward = 0
    while not done:
        action, new_memories = agent.get_action(state, prev_memories, train=False)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        prev_memories = new_memories
        total_reward += reward

        if done:
            break

    return total_reward


def train(make_env, agent, log_folder='logs', name='DRQNROST', epoch_n=500, episode_n=100, test_n=1):
    train_info = TrainLog(name, agent.get_hyper_parameters())
    env = make_env()
    pool = Pool(agent, make_env)

    for epoch in range(epoch_n):
        t = time()
        epoch_rewards = []
        for _ in range(episode_n):
            memories = pool.get_memories()
            batch, rewards = pool.make_step()
            agent.fit_agent(batch, memories)
            epoch_rewards += rewards

        mean_reward = np.mean(epoch_rewards)
        test_rewards = [get_session(agent, env) for _ in range(test_n)]
        test_mean_reward = np.mean(test_rewards)

        std_dev = math.sqrt(np.mean([(r - test_mean_reward) ** 2 for r in test_rewards]))

        epoch_info = EpochLog(time() - t, mean_reward, epoch_rewards, test_mean_reward, test_rewards)
        train_info.add_epoch(epoch_info)

        save_log(train_info, log_folder + '\\' + train_info.name)
        print_log(epoch, mean_reward, time() - t, test_mean_reward, std_dev)

        agent.noise.reduce()


def make_env():
    # env = gym.make("CartPole-v2")
    env = DubinsCar()
    # env = SimpleControlProblem_Discrete()
    return env


def main():
    env = make_env()

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    noise = DiscreteUniformNoise(action_n, threshold_decrease=0.01)
    network = SequentialNetwork(state_dim,
                                [(LayerType.Dense, 64),
                                 (LayerType.LSTM, 64),
                                 (LayerType.Dense, 64),
                                 (LayerType.Dense, action_n)],
                                nn.ReLU())
    agent = DRQNROSTAgent(network, noise, state_dim, action_n, batch_size=16, episode_n=2, learning_rate=1e-3, tau=1e-3)

    train(make_env, agent, 'logs\\DubinsCar', 'DRQNROST_1')


if __name__ == '__main__':
    main()
