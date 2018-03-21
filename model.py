import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, class_based=False, cls2wordnum=None):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        if class_based:
            self.class_based = True
            for class_id, word_num in enumerate(cls2wordnum):
                setattr(self, 'word_decoder'+str(class_id), nn.Linear(nhid, word_num))
            self.ncls = len(cls2wordnum)
            self.class_decoder = nn.Linear(nhid, self.ncls)
        else:
            self.class_based = False
            self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if class_based:
                raise ValueError('When using the tied flag, model shouldn\'t be class-based.')
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights(self.class_based)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self, class_based=False):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

        if class_based:
            for decoder_id in range(self.ncls):
                getattr(self, 'word_decoder' + str(decoder_id)).bias.data.fill_(0)
                getattr(self, 'word_decoder' + str(decoder_id)).weight.data.uniform_(-initrange, initrange)
            self.class_decoder.bias.data.fill_(0)
            self.class_decoder.weight.data.uniform_(-initrange, initrange)
        else:
            self.decoder.bias.data.fill_(0)
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, cls2idxlst=None):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        output_flat = output.view(output.size(0)*output.size(1), output.size(2))
        if self.class_based:
            if self.training:
                cls_decoded = self.class_decoder(output_flat).view(output.size(0), output.size(1), -1)
                word_decoded = []
                for class_id in range(self.ncls):
                    words = cls2idxlst[class_id]
                    if len(words) == 0:
                        word_decoded.append(None)
                        continue
                    words_in_class = output_flat.index_select(0, cls2idxlst[class_id])
                    word_decoded.append(getattr(self, 'word_decoder' + str(class_id))(words_in_class))
                return cls_decoded, word_decoded, hidden
            else:
                word_decoded = []
                cls_decoded = self.class_decoder(output_flat).view(output.size(0), output.size(1), -1)
                for class_id in range(self.ncls):
                    word_decoded.append(getattr(self, 'word_decoder' + str(class_id))(output_flat)
                                        .view(output.size(0), output.size(1), -1))
                return cls_decoded, word_decoded, hidden
        else:
            decoded = self.decoder(output)
            return decoded.view(output.size(0), output.size(1), -1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
