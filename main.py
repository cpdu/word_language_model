# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import defaultdict

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--cls', action='store_true',
                    help='use class-based training')
parser.add_argument('--ncls', type=int, default=20,
                    help='number of classes')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
else:
    assert not args.cuda, "You have no CUDA device, so you shouldn't run with --cuda"


###############################################################################
# Load data
###############################################################################

if args.cls:
    corpus = data.Corpus(args.data, class_num=args.ncls)
else:
    corpus = data.Corpus(args.data)


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.cls:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout,
                                      args.tied, args.cls, cls2wordnum=corpus.dictionary.cls2wordnum)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def list2vriable(lst):
    lst = torch.LongTensor(lst)
    if args.cuda:
        lst = lst.cuda()
    return Variable(lst)

def classify(classes, cindices):
    idxdict = defaultdict(lambda :[])
    cidxdict = defaultdict(lambda :[])
    classes = classes.data.cpu()
    cindices = cindices.data.cpu()
    for i, (cls, cidx) in enumerate(zip(classes, cindices)):
        idxdict[cls].append(i)
        cidxdict[cls].append(cidx)

    cls2idxlst = []
    cls2cidxlst = []
    for cls in range(args.ncls):
        cls2idxlst.append(list2vriable(idxdict[cls]))
        cls2cidxlst.append(list2vriable(cidxdict[cls]))
    return cls2idxlst, cls2cidxlst

def evaluate(data_source, class_based=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    if class_based:
        idx2class = list2vriable(corpus.dictionary.idx2class)
        idx2cidx = list2vriable(corpus.dictionary.idx2cidx)

    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        if class_based:
            tar_classes = idx2class.index_select(0, targets)
            tar_cindices = idx2cidx.index_select(0, targets)
            cls2idxlst, cls2cidxlst = classify(tar_classes, tar_cindices)

            cls_output, word_output, hidden = model(data, hidden)

            total_word_num = cls_output.size()[0] * cls_output.size()[1]
            word_loss = 0
            for cls in range(args.ncls):
                if len(cls2idxlst[cls]) == 0:
                    continue
                selected_output = word_output[cls].view(total_word_num, -1).index_select(0, cls2idxlst[cls])
                cindices = cls2cidxlst[cls]
                word_num = cindices.size()[0]
                word_loss += criterion(selected_output, cindices) * word_num
            cls_loss = criterion(cls_output.view(-1, args.ncls), tar_classes)
            total_loss += (cls_loss.data + word_loss.data / total_word_num) * len(data)
        else:
            output, hidden = model(data, hidden)
            total_loss += criterion(output.view(-1, ntokens), targets).data * len(data)
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

def train(class_based=False):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    if class_based:
        idx2class = list2vriable(corpus.dictionary.idx2class)
        idx2cidx = list2vriable(corpus.dictionary.idx2cidx)
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = model.init_hidden(args.batch_size)

        model.zero_grad()
        if class_based:
            tar_classes = idx2class.index_select(0, targets)
            tar_cindices = idx2cidx.index_select(0, targets)
            cls2idxlst, cls2cidxlst = classify(tar_classes, tar_cindices)

            cls_output, word_output, hidden = model(data, hidden, cls2idxlst)

            word_loss = 0
            for cls in range(args.ncls):
                cindices = cls2cidxlst[cls]
                if len(cindices) == 0:
                    continue
                word_num = cindices.size()[0]
                word_loss += criterion(word_output[cls], cindices) * word_num
            cls_loss = criterion(cls_output.view(-1, args.ncls), tar_classes)
            total_word_num = targets.size()[0]
            loss = cls_loss + word_loss / total_word_num
        else:
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optim.step()

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(args.cls)
        training_time = time.time() - epoch_start_time

        val_loss = evaluate(val_data, args.cls)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, training_time,
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)
