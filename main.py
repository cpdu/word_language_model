# coding: utf-8
import argparse
import math
import os
from multiprocessing import Process
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import data
from model import RNNModel
from loader import TextDataset

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
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--world_size', type=int, default=0)
parser.add_argument('--rank', type=int, default=-1)
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--shared_file', type=str, default="file://"+os.environ['HOME']+"/shared_file")
parser.add_argument('--distributed', action='store_true')
args = parser.parse_args()



###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

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

def train(model, dataloader, criterion, lr, epoch):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.module.init_hidden(args.batch_size // args.world_size) if args.distributed else model.init_hidden(args.batch_size)
    for batch, data in enumerate(dataloader):
        # data, targets = get_batch(train_data, i)
        targets = data[1].t().contiguous().view(-1)
        data = data[0].t().contiguous()
        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()
        data = Variable(data)
        targets = Variable(targets)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = model.module.init_hidden(args.batch_size // args.world_size) if args.distributed else model.init_hidden(args.batch_size)

        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            if (not args.distributed) or dist.get_rank() == dist.get_world_size() - 1:
                cur_loss = total_loss[0] / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(corpus.train) // args.batch_size // args.bptt, lr,
                                  elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

def evaluate(model, criterion, eval_batch_size, data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.module.init_hidden(eval_batch_size) if args.distributed else model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

def main():
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    # construct training dataloader
    dataset = TextDataset(corpus.train, args.bptt)
    if args.distributed:
        dsampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size // args.world_size, sampler=dsampler, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    eval_batch_size = 10
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    # construct model
    model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
    if args.cuda:
        model.cuda()
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, dim=1)

    criterion = nn.CrossEntropyLoss()
    lr = args.lr

    # Loop over epochs.
    # At any point you can hit Ctrl + C to break out of training early.
    best_val_loss = None
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(model, dataloader, criterion, lr, epoch)
            val_loss = evaluate(model, criterion, eval_batch_size, val_data)

            if (not args.distributed) or dist.get_rank() == dist.get_world_size() - 1:
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss, math.exp(val_loss)))
                print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if (not args.distributed) or dist.get_rank() == dist.get_world_size() - 1:
                    with open(args.save, 'wb') as f:
                        torch.save(model.module, f) if args.distributed else torch.save(model, f)
                    best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0

    except KeyboardInterrupt:
        if dist.get_rank() == dist.get_world_size() - 1:
            print('-' * 89)
            print('Exiting from training early')

    # Load the best saved model.
    if (not args.distributed) or dist.get_rank() == dist.get_world_size() - 1:
        with open(args.save, 'rb') as f:
            if args.distributed:
                model.module = torch.load(f)
            else:
                model = torch.load(f)

        # Run on test data.
        test_loss = evaluate(model, criterion, eval_batch_size, test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)

def init_processes(rank, size, func, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    dist.init_process_group(backend, rank=rank, world_size=size, init_method=args.shared_file)
    func()


if __name__ == "__main__":
    if args.distributed:
        if args.batch_size % args.world_size != 0:
            print("\nError: batch_size must be an integer multiple of world_size.\n")
            exit(1)
        if args.world_size < 2:
            print("\nError: world_size should be greater than 1 in distributed mode.\n")
            exit(2)
        p = Process(target=init_processes, args=(args.rank, args.world_size, main))
        p.start()
        p.join()
    else:
        main()