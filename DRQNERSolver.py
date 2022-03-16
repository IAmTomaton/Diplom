import math
from time import time

import numpy as np

from log import save_log
from train_info.epoch_log import EpochLog
from train_info.train_log import TrainLog
from utils import print_log


def get_session(agent, env, train_agent=False):
    state = env.reset()
    agent.reset()
    total_reward = 0
    while True:
        action = agent.get_action(state, train=train_agent)
        next_state, reward, done, _ = env.step(action)
        if train_agent:
            agent.fit_agent(state, action, reward, done, next_state)
        state = next_state
        total_reward += reward

        if done:
            break

    return total_reward


def train(env, agent, log_folder, name, epoch_n=200, session_n=20, test_n=20):
    train_info = TrainLog(name, agent.get_hyper_parameters())

    for epoch in range(epoch_n):
        t = time()
        rewards = [get_session(agent, env, train_agent=True) for _ in range(session_n)]
        agent.noise.reduce()
        mean_reward = np.mean(rewards)

        test_rewards = [get_session(agent, env) for _ in range(test_n)]
        test_mean_reward = np.mean(test_rewards)
        std_dev = math.sqrt(np.mean([(r - test_mean_reward) ** 2 for r in test_rewards]))

        epoch_info = EpochLog(time() - t, mean_reward, rewards, test_mean_reward, test_rewards)
        train_info.add_epoch(epoch_info)

        save_log(train_info, log_folder + '\\' + train_info.name)
        print_log(epoch, mean_reward, time() - t, test_mean_reward, std_dev)
