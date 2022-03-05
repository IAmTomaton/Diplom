import math
import numpy as np
import torch
from torch import nn
from time import time
from train_info.epoch_log import EpochLog
from train_info.train_log import TrainLog
from log import save_log
from networks.NetworkD64D72LSTM64 import NetworkD64D72LSTM64
from other.SimpleControlProblem_Discrete import SimpleControlProblem_Discrete
from utils import print_log


class DRQNROAgent(nn.Module):

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
        self._memories = [self.get_initial_state(1) for _ in range(self.batch_size)]
        self._env_states = [env.reset() for env in self._envs]
        self._rewards = [0 for _ in range(self.batch_size)]
        self._ended_rewards = []

    def get_action(self, memory, state, train=False):
        state = torch.FloatTensor(np.array(state))
        new_memory, readouts = self._q.step(memory, state)
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

        for SARDSes in batch:
            for i in range(self.batch_size):
                state, action, reward, done, next_state = SARDSes[i]
                memory = memories[i]

                state = torch.FloatTensor(np.array([state]))
                lstm_state, q_value = self._q(memory, state)

                next_state = torch.FloatTensor(np.array([next_state]))
                next_lstm_state, next_q_value = self._q(lstm_state, next_state)

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

        return self._ended_rewards, int(loss)

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


def train(env, agent, log_folder='logs', name='DRQNRO', epoch_n=1000, episode_n=200, test_n=100):
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
    # env = DubinsCar()
    env = SimpleControlProblem_Discrete()
    return env


def main():
    use_cuda = torch.cuda.is_available() and False
    device = torch.device('cuda' if use_cuda else 'cpu')
    env = make_env()
    print('Used', device)

    hyper_parameters = {'gamma': 0.95, 'batch_size': 16, 'learning_rate': 1e-4,
                        'min_epsilon': 1e-4, 'mul_epsilon': 0.9996, 'episode_len': 4}

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    network = NetworkD64D72LSTM64(state_dim, action_n)
    agent = DRQNROAgent(network, state_dim, action_n, make_env, hyper_parameters, device)

    train(env, agent, 'logs_SimpleControlProblem_Discrete')


if __name__ == '__main__':
    main()
