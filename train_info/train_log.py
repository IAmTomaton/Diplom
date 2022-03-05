from .epoch_log import EpochLog


class TrainLog:

    def __init__(self, name, hyper_parameters):
        self.name = name
        self.hyper_parameters = hyper_parameters
        self.epochs = []

    def add_epoch(self, epoch):
        self.epochs.append(epoch)

    @classmethod
    def from_dict(cls, dict):
        train_info = cls(dict["name"], dict["hyper_parameters"])
        train_info._apply_dict(dict)
        return train_info

    def _apply_dict(self, dict):
        self.epochs = list(map(EpochLog.from_json, dict["epochs"]))
