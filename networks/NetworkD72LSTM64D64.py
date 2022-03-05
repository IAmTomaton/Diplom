import torch
from torch import nn


class NetworkD72LSTM64D64(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.rnn_hidden_size = 64
        self.rnn_input_size = 72
        self.hid_1 = nn.Linear(input_dim, self.rnn_input_size)
        self.lstm = nn.LSTMCell(self.rnn_input_size, self.rnn_hidden_size)
        self.hid_2 = nn.Linear(self.rnn_hidden_size, self.rnn_hidden_size)
        self.hid_3 = nn.Linear(self.rnn_hidden_size, output_dim)
        self.relu = nn.ReLU()

    def forward(self, prev_state, obs_state):
        hidden = self.hid_1(obs_state)
        hidden = self.relu(hidden)

        h_new, c_new = self.lstm(hidden, prev_state)

        hidden = self.hid_2(h_new)
        hidden = self.relu(hidden)
        hidden = self.hid_3(hidden)

        return (h_new, c_new), hidden

    def get_initial_state(self, batch_size):
        return torch.zeros((batch_size, self.rnn_hidden_size)), torch.zeros((batch_size, self.rnn_hidden_size))

    def step(self, prev_state, obs_t):
        (h, c), l = self.forward(prev_state, obs_t)
        return (h.detach(), c.detach()), l.detach()
