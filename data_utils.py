# coding=utf8
import unicodedata
import json
import re
import random
import torch

from torch.autograd import Variable
from custom_token import *

with open('config.json') as config_file:
    config = json.load(config_file)

USE_CUDA = config['TRAIN']['CUDA']

DATA_PATH = config['DATA']['PATH']
DIALOGUE_CORPUS = config['DATA']['DIALOGUE_CORPUS']
# range of sentence length
MIN_LENGTH = config['LOADER']['MIN_LENGTH']
MAX_LENGTH = config['LOADER']['MAX_LENGTH']
# least word count
MIN_COUNT = config['LOADER']['MIN_COUNT']

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z',.!?]+", r" ", sentence)
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def build_data_loader(batch_size=32):
    pairs = []
    length_range = range(MIN_LENGTH, MAX_LENGTH)
    print('Loading Corpus.')
    with open(DATA_PATH + DIALOGUE_CORPUS) as my_file:
        i = 0
        for line in my_file:
            i += 1
            pa, pb = line[:-1].split(' +++$+++ ')
            pa = pa.split()
            pb = pb.split()
            if len(pa) in length_range and len(pb) in length_range:
                pairs.append((pa, pb))
                # for text
                # i += 1
                # if i >= 3000:
                #     break

    print('Read dialogue pair: %d' % len(pairs))
    vocab = Vocabulary()
    for pa, pb in pairs:
        for word in pa + pb:
            vocab.index_word(word)
    vocab.trim(MIN_COUNT)

    keep_pairs = []
    for pa, pb in pairs:
        keep = True
        for word in pa + pb:
            if word not in vocab.word2count:
                keep = False
                break
        if keep:
            keep_pairs.append((pa, pb))
    n_pairs = len(pairs)
    n_keep_pairs = len(keep_pairs)
    print('Trimmed from %d pairs to %d, %.4f of total' % (n_pairs, n_keep_pairs, float(n_keep_pairs) / n_pairs))
    loader = DataLoader(vocab, keep_pairs, batch_size)
    print('Batch number: %d' % len(loader))
    return loader


class Vocabulary(object):
    def __init__(self):
        self.trimmed = False
        self.reset()

    def reset(self):
        self.word2count = {}
        self.word2index = {"PAD": 0, "GO": 1, "EOS": 2}
        self.index2word = {0: "PAD", 1: "GO", 2: "EOS"}
        self.n_words = 3

    # def index_words(self, sentence):
    #     for word in sentence.split():
    #         self.index_word(word)

    def index_word(self, word):
        if word not in self.word2count:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        for word, times in self.word2count.items():
            if times >= min_count:
                keep_words.append(word)
        n_keep_words = len(keep_words)
        n_src_words = len(self.word2index)
        print('keep words %s / %s = %.4f' % (n_keep_words, n_src_words, float(n_keep_words) / n_src_words))

        self.reset()
        for word in keep_words:
            self.index_word(word)


class DataLoader(object):
    def __init__(self, vocabulary, src_pairs, batch_size):
        self.vocabulary = vocabulary
        self.data = []
        self.test = []

        n_iter = int(len(src_pairs) / batch_size)
        n_test = int(n_iter * 0.1)
        assert (n_iter >= 10)

        src_pairs = sorted(src_pairs, key=lambda s: len(s[1]))

        train_decode_length, test_decoder_length = 0.0, 0.0
        # choose items to put into test set
        test_ids = random.sample(range(n_iter), n_test)
        for i in range(n_iter):
            batch_seq_pairs = sorted(src_pairs[i * batch_size: (i + 1) * batch_size], key=lambda s: len(s[0]),
                                     reverse=True)
            decode_length = float(sum([len(x[1]) for x in batch_seq_pairs])) / 64

            input_group, target_group = self.__process(batch_seq_pairs)
            if i not in test_ids:
                self.data.append((input_group, target_group))
                train_decode_length += decode_length
            else:
                # test data
                self.test.append((input_group, target_group))
                test_decoder_length += decode_length

        self.train_data_len = len(self.data)
        self.test_data_len = len(self.test)
        mean_train_decode_len = train_decode_length / self.train_data_len
        mean_test_decode_len = test_decoder_length / self.test_data_len
        print('mean decode length: (%.2f, %.2f)' % (mean_train_decode_len, mean_test_decode_len))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_vocabulary_size(self):
        return self.vocabulary.n_words

    def random_batch(self):
        # assert(self.train_data_len > 1)
        return self.data[random.randint(0, self.train_data_len - 1)]

    def random_test(self):
        # assert(self.test_data_len > 1)
        return self.test[random.randint(0, self.test_data_len - 1)]

    def __process(self, batch_seq_pairs):
        input_seqs, target_seqs = zip(*batch_seq_pairs)
        # convert to index
        input_seqs = [self.indexes_from_sentence(s) for s in input_seqs]
        target_seqs = [self.indexes_from_sentence(s) for s in target_seqs]
        # PAD input_seqs
        input_lens = [len(s) for s in input_seqs]
        max_input_len = max(input_lens)
        input_padded = [self.pad_seq(s, max_input_len) for s in input_seqs]
        # PAD target_seqs
        target_lens = [len(s) for s in target_seqs]
        max_target_len = max(target_lens)
        target_padded = [self.pad_seq(s, max_target_len) for s in target_seqs]
        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
        target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
        if USE_CUDA:
            input_var = input_var.cuda()
            target_var = target_var.cuda()
        return (input_var, input_lens), (target_var, target_lens)

    def indexes_from_sentence(self, sentence):
        return [self.vocabulary.word2index[word] for word in sentence] + [EOS_token]

    def pad_seq(self, seq, max_length):
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq
