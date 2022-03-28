import math
from time import time

import numpy as np

from log import save_log
from train_info.epoch_log import EpochLog
from train_info.train_log import TrainLog
from utils import print_log


def get_session(env, agent, learning=False, session_len=None):
    state = env.reset()
    agent.reset()
    states, actions, rewards = [], [], []
    n = 0
    while session_len is None or n < session_len:
        n += 1
        states.append(state)

        action = agent.get_action(state)
        actions.append(action)

        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)

        if learning:
            agent.fit(state, action, reward, done, next_state)
        state = next_state

        if done:
            break

    return states, actions, rewards


def go(env, agent, episode_n=20, learning=True, show=None, session_len=None):
    sessions = []

    for episode in range(episode_n):
        states, actions, rewards = get_session(env, agent, learning, session_len)
        session = {'states': states, 'actions': actions, 'rewards': rewards}
        sessions.append(session)

        if learning:
            agent.noise.reduce()

        if show:
            show(env, agent, episode, [session])
    return sessions


def train(env, agent, epoch_n=200, episode_n=10, test_episode_n=10, log=False, log_folder='', name_suffix=''):
    if log:
        train_info = TrainLog(agent.name + name_suffix, agent.get_hyper_parameters())

    for epoch in range(epoch_n):
        start_time = time()
        sessions = go(env, agent, episode_n)
        epoch_time = time() - start_time

        if log:
            rewards = [sum(session['rewards']) for session in sessions]
            mean_reward = np.mean(rewards)

            agent.noise.turn_off()
            test_sessions = go(env, agent, test_episode_n, False)
            agent.noise.turn_on()

            test_rewards = [sum(session['rewards']) for session in test_sessions]
            test_mean_reward = np.mean(test_rewards)
            std_dev = math.sqrt(np.mean([(r - test_mean_reward) ** 2 for r in test_rewards]))

            epoch_info = EpochLog(epoch_time, mean_reward, rewards, test_mean_reward, test_rewards)
            train_info.add_epoch(epoch_info)

            save_log(train_info, log_folder + '\\' + train_info.name)
            print_log(epoch, mean_reward, epoch_time, test_mean_reward, std_dev)
