from enum import Enum
import torch
from torch import nn


class LayerType(Enum):
    LSTM = 'LSTM'
    Dense = 'Dense'


class SequentialNetworkWithTypes(nn.Module):

    def __init__(self, input_dim, layers, hidden_activation, output_activation=None):
        super().__init__()

        self._hidden_activation = hidden_activation
        self._output_activation = output_activation

        self._layer_types = []
        self._layers = nn.ModuleList()
        self._lstm_size = 0

        self._architecture = []

        for layer_type, layer_dim in layers:
            self._architecture.append(layer_type.value + '(' + str(input_dim) + ', ' + str(layer_dim) + ')')
            if layer_type == LayerType.Dense:
                layer = nn.Linear(input_dim, layer_dim)
            elif layer_type == LayerType.LSTM:
                layer = nn.LSTMCell(input_dim, layer_dim)
                self._lstm_size = layer_dim
            else:
                raise Exception('Такого слоя нет ' + layer_type)

            input_dim = layer_dim
            self._layers.append(layer)
            self._layer_types.append(layer_type)

    def forward(self, state, memory=None):
        hid = state
        for i, layer_type in enumerate(self._layer_types):
            layer = self._layers[i]

            if layer_type == LayerType.Dense:
                hid = layer(hid)
            elif layer_type == LayerType.LSTM:
                memory = layer(hid, memory)
                hid = memory[0]

            if i < len(self._layer_types) - 1:
                hid = self._hidden_activation(hid)

        if self._output_activation is not None:
            hid = self._output_activation(hid)

        if memory is None:
            return hid

        return hid, memory

    def get_initial_state(self, batch_size=1):
        return torch.zeros((batch_size, self._lstm_size)), torch.zeros((batch_size, self._lstm_size))

    def step(self, state, memory=None):
        if memory is None:
            l = self.forward(state, memory)
            return l.detach()

        l, (h, c) = self.forward(state, memory)
        return l.detach(), (h.detach(), c.detach())

    def get_hyper_parameters(self):
        return {'layers': self._architecture, 'hidden_activation': str(self._hidden_activation),
                'output_activation': str(self._output_activation)}
