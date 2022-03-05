import json


def save_log(data, path):
    with open(path + '.json', "w") as write_file:
        json.dump(data.__dict__, write_file, default=lambda o: o.__dict__)


def read_log(path):
    with open(path, "r") as read_file:
        data = json.load(read_file)
    return data
