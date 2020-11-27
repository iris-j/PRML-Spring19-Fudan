from SkipRNNCells import *
from torch.nn.init import xavier_uniform_
from fastNLP.core.losses import LossBase
from fastNLP.core.metrics import MetricBase
from BasicRNNNetworks import CBasicLSTMCell, CBasicGRUCell
from BaiscRNNCells import BasicLSTMCell, BasicGRUCell


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


class CCellBaseSkipLSTM(CCellBase):
    def __init__(self, cell, learnable_elements, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, activation=torch.tanh, layer_norm=False):
        super(CCellBaseSkipLSTM, self).__init__(cell, learnable_elements, input_size, hidden_size, num_layers, bias, batch_first, activation, layer_norm)
        self.weight_uh = Parameter(xavier_uniform_(torch.Tensor(1, hidden_size)))
        if bias:
            self.bias_uh = Parameter(torch.ones(1))
        else:
            self.register_parameter('bias_uh', None)

    def forward(self, word_seq, hx=None):
        if len(word_seq.shape) == 3:
            if self.batch_first:
                word_seq = word_seq.transpose(0, 1)
            sequence_length, batch_size, input_size = word_seq.shape
        else:
            sequence_length = 1
            batch_size, input_size = word_seq.shape

        if hx is None:
            hx = self.init_hidden(batch_size)  # 虚函数，子类实现
            if word_seq.is_cuda:
                if self.num_layers == 1:
                    hx = tuple([x.cuda() for x in hx])
                else:
                    hx = [tuple([y.cuda() if y is not None else None for y in x]) for x in hx]


        # Initialize batchnorm layers
        if self.layer_norm and self.lst_bnorm_rnn is None:
            self.lst_bnorm_rnn = []
            for i in np.arange(self.num_layers):
                lst_bnorm_rnn_tmp = torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for i in np.arange(4)])
                if word_seq.is_cuda:
                    lst_bnorm_rnn_tmp = lst_bnorm_rnn_tmp.cuda()
                self.lst_bnorm_rnn.append(lst_bnorm_rnn_tmp)
            self.lst_bnorm_rnn = torch.nn.ModuleList(self.lst_bnorm_rnn)

        lst_output = []
        lst_update_gate = []
        for t in np.arange(sequence_length):
            output, hx = self.cell(word_seq[t], hx, self.num_layers,
                                   self.weight_ih, self.weight_hh, self.weight_uh,
                                   self.bias_ih, self.bias_hh, self.bias_uh,
                                   activation=self.activation, lst_layer_norm=self.lst_bnorm_rnn)
            new_h, update_gate = output
            lst_output.append(new_h)
            lst_update_gate.append(update_gate)

        output = torch.stack(lst_output)
        update_gate = torch.stack(lst_update_gate)

        if self.batch_first:
            output = output.transpose(0, 1)
            update_gate = update_gate.transpose(0, 1)

        return output, hx, update_gate


class CCellBaseSkipGRU(CCellBase):
    def __init__(self, cell, learnable_elements, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, activation=torch.tanh, layer_norm=False):
        super(CCellBaseSkipGRU, self).__init__(cell, learnable_elements, input_size, hidden_size, num_layers, bias, batch_first, activation, layer_norm)
        self.weight_uh = Parameter(xavier_uniform_(torch.Tensor(1, hidden_size)))
        if bias:
            self.bias_uh = Parameter(torch.ones(1))
        else:
            self.register_parameter('bias_uh', None)

    def forward(self, word_seq, hx=None):
        if len(word_seq.shape) == 3:
            if self.batch_first:
                word_seq = word_seq.transpose(0, 1)
            sequence_length, batch_size, input_size = word_seq.shape
        else:
            sequence_length = 1
            batch_size, input_size = word_seq.shape

        if hx is None:
            hx = self.init_hidden(batch_size)  # 虚函数，子类实现
            if word_seq.is_cuda:
                if self.num_layers == 1:
                    hx = tuple([x.cuda() for x in hx])
                else:
                    hx = [tuple([y.cuda() if y is not None else None for y in x]) for x in hx]

        # Initialize batchnorm layers
        if self.layer_norm and self.lst_bnorm_rnn is None:
            self.lst_bnorm_rnn = []
            for i in np.arange(self.num_layers):
                lst_bnorm_rnn_tmp = torch.nn.ModuleList(
                    [nn.BatchNorm1d(self.hidden_size) for i in np.arange(4)])
                if word_seq.is_cuda:
                    lst_bnorm_rnn_tmp = lst_bnorm_rnn_tmp.cuda()
                self.lst_bnorm_rnn.append(lst_bnorm_rnn_tmp)
            self.lst_bnorm_rnn = torch.nn.ModuleList(self.lst_bnorm_rnn)

        lst_output = []
        lst_update_gate = []
        for t in np.arange(sequence_length):
            output, hx = self.cell(
                word_seq[t], hx, self.num_layers,
                self.weight_ih, self.weight_hh, self.weight_uh,
                self.bias_ih, self.bias_hh, self.bias_uh,
                activation=self.activation,
                lst_layer_norm=self.lst_bnorm_rnn
            )
            new_h, update_gate = output
            lst_output.append(new_h)
            lst_update_gate.append(update_gate)
        output = torch.stack(lst_output)
        update_gate = torch.stack(lst_update_gate)
        if self.batch_first:
            output = output.transpose(0, 1)
            update_gate = update_gate.transpose(0, 1)
        return output, hx, update_gate


class CSkipGRUCell(CCellBaseSkipGRU):
    def __init__(self, cell, learnable_elements, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, activation=torch.tanh, layer_norm=False):
        super(CSkipGRUCell,self).__init__(cell=SkipGRUCell, learnable_elements=3, input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def init_hidden(self, batch_size):
        return (Variable(torch.randn(batch_size, self.hidden_size)),
                Variable(torch.ones(batch_size, 1), requires_grad=False),
                Variable(torch.zeros(batch_size, 1), requires_grad=False))


class CMultiSkipGRUCell(CCellBaseSkipGRU):
    def __init__(self, cell, learnable_elements, input_size, hidden_size, num_layers, bias=True, batch_first=False, activation=torch.tanh, layer_norm=False):
        super(CMultiSkipGRUCell, self).__init__(cell=MultiSkipGRUCell, learnable_elements=3, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def init_hidden(self, batch_size):
        initial_states = []
        for i in np.arange(self.num_layers):
            initial_h = Variable(torch.randn(batch_size, self.hidden_size))
            if i == self.num_layers - 1:
                initial_update_prob = Variable(torch.ones(batch_size, 1), requires_grad=False)
                initial_cum_update_prob = Variable(torch.zeros(batch_size, 1), requires_grad=False)
            else:
                initial_update_prob = None
                initial_cum_update_prob = None
            initial_states.append((initial_h, initial_update_prob, initial_cum_update_prob))
        return initial_states

class CSkipLSTMCell(CCellBaseSkipLSTM):
    def __init__(self, cell, learnable_elements, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, activation=torch.tanh, layer_norm=False):
        super(CSkipLSTMCell, self).__init__(cell=SkipLSTMCell, learnable_elements=4, input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def init_hidden(self, batch_size):
        return (Variable(torch.randn(batch_size, self.hidden_size)),
                Variable(torch.randn(batch_size, self.hidden_size)),
                Variable(torch.ones(batch_size, 1), requires_grad=False),
                Variable(torch.zeros(batch_size, 1), requires_grad=False))


class CMultiSkipLSTMCell(CCellBaseSkipLSTM):
    def __init__(self, cell, learnable_elements, input_size, hidden_size, num_layers, bias=True, batch_first=False, activation=torch.tanh, layer_norm=False):
        super(CMultiSkipLSTMCell, self).__init__(cell=MultiSkipLSTMCell, learnable_elements=4, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def init_hidden(self, batch_size):
        initial_states = []
        for i in np.arange(self.num_layers):
            initial_c = Variable(torch.randn(batch_size, self.hidden_size))
            initial_h = Variable(torch.randn(batch_size, self.hidden_size))
            if i == self.num_layers - 1:
                initial_update_prob = Variable(torch.ones(batch_size, 1))
                initial_cum_update_prob = Variable(torch.zeros(batch_size, 1))
            else:
                initial_update_prob = None
                initial_cum_update_prob = None
            initial_states.append((initial_c, initial_h, initial_update_prob, initial_cum_update_prob))
        return initial_states


def split_rnn_outputs(model, rnn_outputs):
    if len(rnn_outputs) == 3:
        return rnn_outputs[0], rnn_outputs[1], rnn_outputs[2]
    else:
        return rnn_outputs[0], rnn_outputs[1], None


class SkipBudgetLoss(LossBase):
    def __init__(self, pred=None, target=None, updated_states=None, padding_idx=-100):
        super(SkipBudgetLoss, self).__init__()
        self._init_param_map(pred=pred, target=target, updated_states=updated_states)
        self.padding_idx = padding_idx

    def get_loss(self, pred, target, updated_states):
        res = F.cross_entropy(input=pred, target=target,
                               ignore_index=self.padding_idx)+compute_skip_budget_loss(updated_states)
        return res[0]


def compute_skip_budget_loss(updated_states, cost_per_sample=0.):
    if updated_states is not None:
        return torch.mean(torch.sum(cost_per_sample * updated_states, 1), 0)
    else:
        return torch.zeros(1)


def compute_used_samples(update_state_gate):
    """
    Compute number of used samples (i.e. number of updated states)
    :param update_state_gate: values for the update state gate
    :return: number of used samples
    """
    # 已经除过batch_size
    return update_state_gate.sum() / update_state_gate.shape[0]


class UsedStepsMetric(MetricBase):
    def __init__(self, updated_states=None, sequence_length=None, batch_size=128):
        super().__init__()
        self._init_param_map(updated_states=updated_states, sequence_length=sequence_length)
        self.total = 0
        self.update_count = 0

    def evaluate(self, updated_states, sequence_length, batch_size=128):
        if updated_states is not None:
            self.update_count += compute_used_samples(updated_states).data.cpu().numpy() / sequence_length
        else:
            self.update_count += 1.0
        self.total += 1

    def get_metric(self, reset=True):
        evaluate_result = {'updated_steps_fraction': self.update_count / self.total}
        if reset:
            self.update_count = 0
            self.total = 0
        return evaluate_result


def create_model(model, input_size, hidden_size, num_layers):
    if model == 'skip_lstm':
        if num_layers == 1:
            cells = CSkipLSTMCell(cell=SkipLSTMCell, learnable_elements=4, input_size=input_size, hidden_size=hidden_size, batch_first=True)
        else:
            cells = CMultiSkipLSTMCell(cell=MultiSkipLSTMCell, learnable_elements=4, input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
    if model == 'skip_gru':
        if num_layers == 1:
            cells = CSkipGRUCell(cell=SkipGRUCell, learnable_elements=3, input_size=input_size, hidden_size=hidden_size, batch_first=True)
        else:
            cells = CMultiSkipGRUCell(cell=MultiSkipGRUCell, learnable_elements=3, input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
    if model == 'basic_lstm':
        if num_layers == 1:
            cells = CBasicLSTMCell(cell=BasicLSTMCell, learnable_elements=4, input_size=input_size, hidden_size=hidden_size, batch_first=True)
    if model == 'basic_gru':
        if num_layers == 1:
            cells = CBasicGRUCell(cell=BasicGRUCell, learnable_elements=3, input_size=input_size, hidden_size=hidden_size, batch_first=True)
    return cells
