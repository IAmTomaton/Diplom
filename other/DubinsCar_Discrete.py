import numpy as np


class ActionSpace:

    def __init__(self, n):
        self.n = n


class ObservationSpace:

    def __init__(self, n):
        self.shape = [n]


class DubinsCar:
    def __init__(self, initial_state=np.array([0, 0, 0, 0]), dt=0.1, terminal_time=2 * np.pi, inner_step_n=20,
                 action_values=np.array([[-0.5], [0], [0.5], [1]])):
        self.state_dim = 4
        self.action_values = action_values
        self.action_n = self.action_values.size
        self.terminal_time = terminal_time
        self.dt = dt
        self.initial_state = initial_state
        self.inner_step_n = inner_step_n
        self.inner_dt = dt / inner_step_n

        self.action_space = ActionSpace(len(action_values))
        self.observation_space = ObservationSpace(len(initial_state))

    def reset(self):
        self.state = self.initial_state
        return self.state

    def step(self, action):
        _action = self.action_values[action]
        
        for _ in range(self.inner_step_n):
            self.state = self.state + np.array([1, np.cos(self.state[3]), np.sin(self.state[3]), _action[0]]) * self.inner_dt

        done = False
        reward = 0
        if self.state[0] >= self.terminal_time:
            reward -= np.abs(self.state[1] - 4) + np.abs(self.state[2]) + np.abs(self.state[3] - 0.75 * np.pi)
            done = True

        return self.state, reward, done, None
