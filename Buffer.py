import random


class Buffer:

    def __init__(self, size):
        self._buffer = []
        self._size = size

    def add(self, data):
        self._buffer.append(data)
        if len(self._buffer) > self._size:
            self._buffer.pop(0)

    def get_batch(self, batch_size, batch_len):
        if len(self._buffer) < batch_len:
            raise ValueError("Attempt to take more elements ({0}) than there are in the buffer ({1})"
                             .format(batch_len, len(self._buffer)))
        batch_indexes = random.sample(range(len(self._buffer) - batch_len - 1), batch_size)
        batch = [list(map(lambda j: self._buffer[j + i], batch_indexes)) for i in range(batch_len)]
        return batch

    def __len__(self):
        return len(self._buffer)
