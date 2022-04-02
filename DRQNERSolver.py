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
    for episode in range(episode_n):
        states, actions, rewards = get_session(env, agent, learning, session_len)

        if learning:
            agent.noise.reduce()

        if show:
            show(env, agent, episode, [{'states': states, 'actions': actions, 'rewards': rewards}])
