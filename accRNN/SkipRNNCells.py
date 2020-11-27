import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.nn import Parameter
import numpy as np


class BinaryLayer(Function):  # binarize the input value
    # 记录operation历史，定义微分公式，通过调用每个Function对象的backward，
    # 将返回的梯度传给下一个Function, 和Function交互的唯一方法是重写Function的子类
    # 所有Function的子类都要重写backward和forward方法

    def forward(self, input):
        return input.round()  # 四舍五入

    def backward(self, grad_output):
        return grad_output


def SkipLSTMCell(input, hidden, num_layers, w_ih, w_hh, w_uh, b_ih=None, b_hh=None, b_uh=None, activation=torch.tanh, lst_layer_norm=None):
    w_ih, w_hh = w_ih[0], w_hh[0]
    b_ih = b_ih[0] if b_ih else None
    b_hh = b_hh[0] if b_hh else None

    c_pre, h_pre, update_prob_pre, cum_update_prob_pre = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(h_pre, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    if lst_layer_norm:  # channel方向做归一化，算CHW的均值，主要对RNN作用明显；
        ingate = lst_layer_norm[0][0](ingate.contiguous())  # tensor在内存中连续分布
        forgetgate = lst_layer_norm[0][1](forgetgate.contiguous())
        cellgate = lst_layer_norm[0][2](cellgate.contiguous())
        outgate = lst_layer_norm[0][3](outgate.contiguous())

    # update gate for timestamp t
    bn = BinaryLayer()
    test = 1. - cum_update_prob_pre

    cum_update_prob = cum_update_prob_pre + torch.min(update_prob_pre, test)
    update_gate = bn(cum_update_prob)

    if update_gate[0] == 0:
        new_c = c_pre
        new_h = h_pre
        new_update_prob = update_prob_pre
        new_cum_update_prob = cum_update_prob

    else:
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = activation(cellgate)
        outgate = torch.sigmoid(outgate)
        new_c = forgetgate * c_pre + ingate * cellgate
        new_h = outgate * activation(new_c)
        new_update_prob = torch.sigmoid(F.linear(new_c, w_uh, b_uh))
        new_cum_update_prob = update_gate * 0.

    new_state = (new_c, new_h, new_update_prob, new_cum_update_prob)
    new_output = (new_h, update_gate)

    return new_output, new_state


def MultiSkipLSTMCell(input, state, num_layers, w_ih, w_hh, w_uh, b_ih=None, b_hh=None, b_uh=None, activation=torch.tanh, lst_layer_norm=None):
    _, _, update_prob_pre, cum_update_prob_pre = state[-1]
    cell_input = input
    state_candidates = []
    new_states = []
    # update gate for timestamp t
    bn = BinaryLayer()
    cum_update_prob = cum_update_prob_pre + torch.min(update_prob_pre, 1. - cum_update_prob_pre)
    update_gate = bn(cum_update_prob)

    if update_gate[0] == 0:
        for idx in np.arange(num_layers - 1):
            new_c = state[idx][0]
            new_h = state[idx][1]
            new_states.append((new_c, new_h, None, None))
        new_c = state[-1][0]
        new_h = state[-1][1]
        new_update_prob = update_prob_pre
        new_cum_update_prob = cum_update_prob

    else:
        for idx in np.arange(num_layers):
            c_pre, h_pre, _, _ = state[idx]
            gates = F.linear(cell_input, w_ih[idx], b_ih[idx]) + F.linear(h_pre, w_hh[idx], b_hh[idx])

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            if lst_layer_norm:
                ingate = lst_layer_norm[idx][0](ingate.contiguous())
                forgetgate = lst_layer_norm[idx][1](forgetgate.contiguous())
                cellgate = lst_layer_norm[idx][2](cellgate.contiguous())
                outgate = lst_layer_norm[idx][3](outgate.contiguous())
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = activation(cellgate)
            outgate = torch.sigmoid(outgate)
            new_c_t = forgetgate * c_pre + ingate * cellgate
            new_h_t = outgate * activation(new_c_t)
            state_candidates.append((new_c_t, new_h_t))
            cell_input = new_h_t

        for idx in np.arange(num_layers - 1):
            new_c = state_candidates[idx][0]
            new_h = state_candidates[idx][1]
            new_states.append((new_c, new_h, None, None))
        new_c = state_candidates[-1][0]
        new_h = state_candidates[-1][1]
        new_update_prob = F.sigmoid(F.linear(state_candidates[-1][0], w_uh, b_uh))
        new_cum_update_prob = update_gate * 0.

    new_states.append((new_c, new_h, new_update_prob, new_cum_update_prob))
    new_output = (new_h, update_gate)
    return new_output, new_states


def SkipGRUCell(input, hidden, num_layers, w_ih, w_hh, w_uh, b_ih=None, b_hh=None, b_uh=None,
                activation=F.tanh, lst_layer_norm=None):
    w_ih, w_hh = w_ih[0], w_hh[0]
    b_ih = b_ih[0] if b_ih else None
    b_hh = b_hh[0] if b_hh else None

    h_pre, update_prob_pre, cum_update_prob_pre = hidden

    # update gate for timestamp t
    bn = BinaryLayer()
    cum_update_prob = cum_update_prob_pre + torch.min(update_prob_pre, 1. - cum_update_prob_pre)
    update_gate = bn(cum_update_prob)

    if update_gate[0] == 0:
        new_h = h_pre
        new_update_prob = update_prob_pre
        new_cum_update_prob = cum_update_prob

    else:
        gi = F.linear(input, w_ih, b_ih)
        gh = F.linear(h_pre, w_hh, b_hh)
        i_reset, i_input, i_new = gi.chunk(3, 1)
        h_reset, h_input, h_new = gh.chunk(3, 1)

        resetgate_t = i_reset + h_reset
        inputgate_t = i_input + h_input
        if lst_layer_norm:
            resetgate_t = lst_layer_norm[0][0](resetgate_t.contiguous())
            inputgate_t = lst_layer_norm[0][1](inputgate_t.contiguous())

        resetgate = F.sigmoid(resetgate_t)
        inputgate = F.sigmoid(inputgate_t)
        newgate = activation(i_new + resetgate * h_new)

        new_h = (1. - inputgate) * h_pre + inputgate * newgate
        new_update_prob = F.sigmoid(F.linear(new_h, w_uh, b_uh))
        new_cum_update_prob = update_gate * 0.

    new_state = (new_h, new_update_prob, new_cum_update_prob)
    new_output = (new_h, update_gate)

    return new_output, new_state


def MultiSkipGRUCell(input, hidden, num_layers, w_ih, w_hh, w_uh, b_ih=None, b_hh=None, b_uh=None, activation=torch.tanh, lst_layer_norm=None):
    _, update_prob_pre, cum_update_prob_pre = hidden[-1]
    cell_input = input
    state_candidates = []
    new_states = []

    # update gate for timestamp t
    bn = BinaryLayer()
    cum_update_prob = cum_update_prob_pre + torch.min(update_prob_pre, 1. - cum_update_prob_pre)
    update_gate = bn(cum_update_prob)

    if update_gate[0] == 0:
        for idx in np.arange(num_layers - 1):
            new_h = hidden[idx][0]
            new_states.append((new_h, None, None))
        new_h = hidden[-1][0]
        new_update_prob = update_prob_pre
        new_cum_update_prob = cum_update_prob

    else:
        for idx in np.arange(num_layers):
            h_pre, _, _ = hidden[idx]
            gi = F.linear(cell_input, w_ih[idx], b_ih[idx])
            gh = F.linear(h_pre, w_hh[idx], b_hh[idx])
            i_reset, i_input, i_new = gi.chunk(3, 1)
            h_reset, h_input, h_new = gh.chunk(3, 1)

            resetgate_t = i_reset + h_reset
            inputgate_t = i_input + h_input

            if lst_layer_norm:
                resetgate_t = lst_layer_norm[idx][0](resetgate_t.contiguous())
                inputgate_t = lst_layer_norm[idx][1](inputgate_t.contiguous())

            resetgate = F.sigmoid(resetgate_t)
            inputgate = F.sigmoid(inputgate_t)
            newgate = activation(i_new + resetgate * h_new)
            new_h = (1. - inputgate) * h_pre + inputgate * newgate
            state_candidates.append(new_h)
            cell_input = new_h

        for idx in np.arange(num_layers - 1):
            new_h = state_candidates[idx]
            new_states.append((new_h, None, None))
        new_h = state_candidates[-1]
        new_update_prob = F.sigmoid(F.linear(state_candidates[-1], w_uh, b_uh))
        new_cum_update_prob = update_gate * 0.

    new_states.append((new_h, new_update_prob, new_cum_update_prob))
    new_output = (new_h, update_gate)

    return new_output, new_states



