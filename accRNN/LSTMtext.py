from Text_data_processing import *
from SkipRNNNetworks import *
from fastNLP import Adam, CrossEntropyLoss
from fastNLP import AccuracyMetric
from fastNLP import Trainer, Tester
import time
import fitlog
from fastNLP.core.callback import FitlogCallback

fitlog.commit(__file__)
fitlog.add_hyper_in_file(__file__)

# hypers
model_name = 'skip_lstm'
task = 'text_classification'
hidden_units = 128
num_layers = 1
batch_size = 32
learning_rate = 1e-3
# hypers

fitlog.add_hyper({'model_name': model_name, 'task': task, 'hidden_units': hidden_units, 'num_layers': num_layers, 'batch_size': batch_size, 'learning_rate': learning_rate})


class TextModel(nn.Module):
    def __init__(self, cells, model, embed_num, embed_dim, hidden_dim, output_dim, pre_weight=None):
        super(TextModel, self).__init__()
        self.model = model
        self.rnn = cells
        if pre_weight is None:
            self.embed = nn.Embedding(embed_num, embed_dim)
        else:
            self.embed = nn.Embedding.from_pretrained(pre_weight)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, word_seq, hx=None):
        x = self.embed(word_seq)
        if hx is not None:
            output = self.rnn(x, hx)
        else:
            output = self.rnn(x)
        output, hx, updated_state = split_rnn_outputs(self.model, output)
        # little modification, feeding the average of the outputs at different time steps to the full-connect layer
        output = torch.mean(output, dim=1, keepdim=True)
        output = self.fc(output[:, -1, :])
        return {'pred': output, 'updated_states': updated_state, 'sequence_length': x.shape[1]}

    def predict(self, word_seq):
        output = self(word_seq)
        _, predict = output['pred'].max(dim=1)
        return {'pred': predict, 'updated_states': output['updated_states'],
                'sequence_length': output['sequence_length']}  # 用于metric


train_data, test_data, vocab = get_fastnlp_dataset()
cells = create_model(model_name, input_size=128, hidden_size=hidden_units, num_layers=num_layers)

mymodel = TextModel(cells, model=model_name, embed_num=len(vocab), embed_dim=128, hidden_dim=hidden_units, output_dim=8)
# mymodel = load_model(mymodel, './model/best_TextModel_acc_2019-06-28-09-07-50')
trainer = Trainer(train_data=train_data, model=mymodel,
                  loss=CrossEntropyLoss(pred='pred', target='target'),
                  # loss=SkipBudgetLoss(pred='pred', target='target', updated_states='updated_states'),
                  metrics=[AccuracyMetric(), UsedStepsMetric()],
                  n_epochs=30,
                  batch_size=batch_size,
                  print_every=-1,
                  validate_every=-1,
                  dev_data=test_data,
                  save_path='./model',
                  optimizer=Adam(lr=learning_rate, weight_decay=0),
                  check_code_level=0,
                  device="cuda",
                  metric_key='acc',
                  use_tqdm=False,
                  callbacks=[FitlogCallback(test_data)])
start = time.clock()
trainer.train()
end = time.clock()
training_time = end - start
print('total training time:%fs' % (end - start))
fitlog.add_hyper({'time': training_time})

fitlog.finish()
