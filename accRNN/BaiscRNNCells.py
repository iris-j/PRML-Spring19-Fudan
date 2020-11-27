import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable, Function


def BasicLSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None,
                  activation=torch.tanh, lst_layer_norm=None):

    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    if lst_layer_norm:
        ingate = lst_layer_norm[0](ingate.contiguous())
        forgetgate = lst_layer_norm[1](forgetgate.contiguous())
        cellgate = lst_layer_norm[2](cellgate.contiguous())
        outgate = lst_layer_norm[3](outgate.contiguous())

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = activation(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * activation(cy)

    return hy, cy


def BasicGRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None,
                  activation=torch.tanh, lst_layer_norm=None):

    gi = F.linear(input, w_ih, b_ih)
    gh = F.linear(hidden, w_hh, b_hh)
    i_reset, i_input, i_new = gi.chunk(3, 1)
    h_reset, h_input, h_new = gh.chunk(3, 1)

    resetgate_t = i_reset + h_reset
    inputgate_t = i_input + h_input
    if lst_layer_norm:
        resetgate_t = lst_layer_norm[0](resetgate_t.contiguous())
        inputgate_t = lst_layer_norm[1](inputgate_t.contiguous())

    resetgate = torch.sigmoid(resetgate_t)
    inputgate = torch.sigmoid(inputgate_t)
    newgate = activation(i_new + resetgate * h_new)
    hy = (1. - inputgate) * hidden + inputgate * newgate

    return hy
