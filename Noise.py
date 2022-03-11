import numpy as np


class DiscreteUniformNoise:
    def __init__(self, action_n, threshold=1, threshold_min=1e-2, threshold_decrease=1e-2):
        self.action_n = action_n
        self.start_threshold = threshold
        self.threshold = threshold
        self.threshold_min = threshold_min
        self.threshold_decrease = threshold_decrease

    def get(self):
        return np.random.choice(self.action_n)

    def reduce(self):
        if self.threshold > self.threshold_min:
            self.threshold -= self.threshold_decrease

    def reset(self):
        self.threshold = self.start_threshold

    def get_hyper_parameters(self):
        return {'threshold': self.start_threshold, 'threshold_min': self.threshold_min,
                'threshold_decrease': self.threshold_decrease}
