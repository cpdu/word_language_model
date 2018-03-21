import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word2freq = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2freq[word] = 0
            self.word2idx[word] = len(self.idx2word) - 1
        self.word2freq[word] += 1
        return self.word2idx[word]

    def classify(self, text_len, class_num):
        """ classify words into classes, trying to keep the sum of frequency over all words in each class equal """
        assert text_len > class_num, "The number of classes cannot be bigger than text size."
        # sort words in descending order according to its frequency in texts
        freq_desc = sorted(self.word2freq.items(), key=lambda x: x[1], reverse=True)

        freq_per_class = text_len * 1. / class_num
        freq_threshold = freq_per_class
        class_freq = 0
        class_id = 0
        word_num = 0
        innerdict = {}
        self.idx2cidx = [0] * len(self)
        self.idx2class = [0] * len(self)
        self.cls2wordnum = []
        self.lst_cidx2idx = []

        # classify words into classes
        for word, word_freq in freq_desc:
            class_freq += word_freq
            self.idx2class[self.word2idx[word]] = class_id
            self.idx2cidx[self.word2idx[word]] = word_num
            innerdict[word_num] = self.word2idx[word]
            word_num += 1
            if class_freq > freq_threshold or class_freq == text_len:
                self.cls2wordnum.append(word_num)
                self.lst_cidx2idx.append(innerdict)
                innerdict.clear()
                word_num = 0
                class_id += 1
                freq_threshold += freq_per_class

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, class_num=0):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        if class_num != 0:
            text_len = len(self.train) + len(self.valid) + len(self.test)
            self.dictionary.classify(text_len, class_num)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

if __name__=='__main__':
    corpus = Corpus('./data/wikitext-2')