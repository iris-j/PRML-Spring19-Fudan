import torch
import torch.nn as nn
import numpy as np
from BaiscRNNCells import *
from torch.nn.init import xavier_uniform_
from torch.nn import Parameter


class CCellBase(nn.Module):
    def __init__(self, cell, learnable_elements, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, activation=torch.tanh, layer_norm=False):
        super(CCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.cell = cell
        self.num_layers = num_layers
        self.weight_ih = []
        self.weight_hh = []
        self.bias_ih = []
        self.bias_hh = []

        # init parameters
        for i in np.arange(self.num_layers):
            if i == 0:
                weight_ih = Parameter(xavier_uniform_(torch.Tensor(learnable_elements * hidden_size, input_size)))
            else:
                weight_ih = Parameter(xavier_uniform_(torch.Tensor(learnable_elements * hidden_size, hidden_size)))

            weight_hh = Parameter(xavier_uniform_(torch.Tensor(learnable_elements * hidden_size, hidden_size)))
            self.weight_ih.append(weight_ih)
            self.weight_hh.append(weight_hh)
            if bias:
                bias_ih = Parameter(torch.zeros(learnable_elements*hidden_size))
                bias_hh = Parameter(torch.zeros(learnable_elements*hidden_size))
                self.bias_ih.append(bias_ih)
                self.bias_hh.append(bias_hh)
            else:
                self.register_parameter('bias_ih_'+str(i), None)
                self.register_parameter('bias_hh_'+str(i), None)

            self.weight_ih = nn.ParameterList(self.weight_ih)
            self.weight_hh = nn.ParameterList(self.weight_hh)

            if self.bias_ih:
                self.bias_ih = nn.ParameterList(self.bias_ih)
                self.bias_hh = nn.ParameterList(self.bias_hh)

            self.activation = activation
            self.layer_norm = layer_norm
            self.lst_bnorm_rnn = None


class CCellBaseLSTM(CCellBase):
    def forward(self, word_seq, hx=None):
        if len(word_seq.shape) == 3:
            if self.batch_first:
                word_seq = word_seq.transpose(0, 1)
            sequence_length, batch_size, input_size = word_seq.shape
        else:
            sequence_length = 1
            batch_size, input_size = word_seq.shape

        if hx is None:
            hx = self.init_hidden(batch_size)
            if word_seq.is_cuda:
                hx = tuple([x.cuda() for x in hx])

        if self.layer_norm and self.lst_bnorm_rnn is None:
            self.lst_bnorm_rnn = torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for i in np.arange(4)])
            if word_seq.is_cuda:
                self.lst_bnorm_rnn = self.lst_bnorm_rnn.cuda()

        lst_output = []
        for t in np.arange(sequence_length):
            hx = self.cell(word_seq[t], hx, self.weight_ih[0], self.weight_hh[0], self.bias_ih[0], self.bias_hh[0],
                           activation=self.activation, lst_layer_norm=self.lst_bnorm_rnn)
            lst_output.append(hx[0])
        output = torch.stack(lst_output)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, hx


class CCellBaseGRU(CCellBase):
    def forward(self, word_seq, hx=None):
        if len(word_seq.shape) == 3:
            if self.batch_first:
                word_seq = word_seq.transpose(0, 1)
            sequence_length, batch_size, input_size = word_seq.shape
        else:
            sequence_length = 1
            batch_size, input_size = word_seq.shape

        if hx is None:
            hx = self.init_hidden(batch_size)
            if word_seq.is_cuda:
                hx = hx.cuda()

        if self.layer_norm and self.lst_bnorm_rnn is None:
            self.lst_bnorm_rnn = torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for i in np.arange(2)])
            if word_seq.is_cuda:
                self.lst_bnorm_rnn = self.lst_bnorm_rnn.cuda()

        lst_output = []
        for t in np.arange(sequence_length):
            hx = self.cell(word_seq[t], hx, self.weight_ih[0], self.weight_hh[0], self.bias_ih[0], self.bias_hh[0],
                           activation=self.activation, lst_layer_norm=self.lst_bnorm_rnn)
            lst_output.append(hx)
        output = torch.stack(lst_output)
        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hx


class CBasicLSTMCell(CCellBaseLSTM):
    def __init__(self, cell, learnable_elements, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, activation=torch.tanh, layer_norm=False):
        super(CBasicLSTMCell, self).__init__(cell=BasicLSTMCell,learnable_elements=4, input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def init_hidden(self, batch_size):
        return (Variable(torch.randn(batch_size, self.hidden_size)),
                Variable(torch.randn(batch_size, self.hidden_size)))


class CBasicGRUCell(CCellBaseGRU):
    def __init__(self, cell, learnable_elements, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
                 activation=torch.tanh, layer_norm=False):
        super(CBasicGRUCell, self).__init__(cell=BasicGRUCell, learnable_elements=3, input_size=input_size,
                                             hidden_size=hidden_size, num_layers=1, batch_first=True)

    def init_hidden(self, batch_size):
        return (Variable(torch.randn(batch_size, self.hidden_size)))
