class EpochLog:

    def __init__(self, time, train_mean, train_rewards, test_mean, test_rewards):
        self.time = time
        self.train_mean = train_mean
        self.train_rewards = train_rewards
        self.test_mean = test_mean
        self.test_rewards = test_rewards

    @classmethod
    def from_json(cls, data):
        return cls(**data)
