# coding=utf8
import json
import os
import random
import re

import torch

from custom_token import *

with open('config.json') as config_file:
    config = json.load(config_file)

DEVICE = torch.device(config['TRAIN']['DEVICE'])

DATA_PATH = config['DATA']['PATH']
DIALOGUE_CORPUS = config['DATA']['DIALOGUE_CORPUS']
# range of sentence length
MIN_LENGTH = config['LOADER']['MIN_LENGTH']
MAX_LENGTH = config['LOADER']['MAX_LENGTH']
# least word count
MIN_COUNT = config['LOADER']['MIN_COUNT']

# Regular expressions used to tokenize.
_WORD_SPLITTER = re.compile(r"([.,!?\"':;)(])")


def basic_tokenizer(sentence):
    """
    Very basic tokenizer: split the sentence into a list of tokens.
    """
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z',.!?]+", r" ", sentence)
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLITTER, space_separated_fragment))
    return [word for word in words if word]


def build_data_loader(batch_size=32):
    couples = []
    length_range = range(MIN_LENGTH, MAX_LENGTH)
    print('Loading Corpus.')
    with open(os.path.join(os.path.abspath(DATA_PATH), DIALOGUE_CORPUS)) as conversation_file:
        i = 0
        for line in conversation_file:
            i += 1
            sentence_a, sentence_b = line[:-1].split(' +++$+++ ')
            sentence_a = sentence_a.split()
            sentence_b = sentence_b.split()
            if len(sentence_a) in length_range and len(sentence_b) in length_range:
                couples.append((sentence_a, sentence_b))

    print('Read dialogue couple: %d' % len(couples))
    vocabulary = Vocabulary()
    for sentence_a, sentence_b in couples:
        for word in sentence_a + sentence_b:
            vocabulary.index_word(word)
    vocabulary.trim(MIN_COUNT)

    valid_couples = []
    for sentence_a, sentence_b in couples:
        valid = True
        for word in sentence_a + sentence_b:
            if word not in vocabulary.word2count:
                valid = False
                break
        if valid:
            valid_couples.append((sentence_a, sentence_b))
    num_src_couples = len(couples)
    num_valid_couples = len(valid_couples)
    print('Trimmed from %d couples to %d, %.4f of total' % (
        num_src_couples, num_valid_couples, float(num_valid_couples) / num_src_couples))
    loader = DataLoader(vocabulary, valid_couples, batch_size)
    print('Batch number: %d' % len(loader))
    return loader


class Vocabulary(object):
    word2count: dict
    word2index: dict
    index2word: dict
    num_words: int

    def __init__(self):
        self.trimmed = False
        self.reset()

    def reset(self):
        self.word2count = {}
        self.word2index = {"PAD": 0, "GO": 1, "EOS": 2}
        self.index2word = {0: "PAD", 1: "GO", 2: "EOS"}
        self.num_words = 3

    # def index_words(self, sentence):
    #     for word in sentence.split():
    #         self.index_word(word)

    def index_word(self, word):
        if word not in self.word2count:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.word2count[word] = 1
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        valid_words = []
        for word, frequency in self.word2count.items():
            if frequency >= min_count:
                valid_words.append(word)
        num_src_words = len(self.word2index)
        num_valid_words = len(valid_words)
        print('Trimmed from %d words to %d, %.4f of total' % (
            num_valid_words, num_src_words, float(num_valid_words) / num_src_words))

        self.reset()
        for word in valid_words:
            self.index_word(word)


class DataLoader(object):
    def __init__(self, vocabulary, src_couples, batch_size):
        self.vocabulary = vocabulary
        self.data = []
        self.test = []

        num_iteration = len(src_couples) // batch_size
        num_test = num_iteration // 10
        assert (num_iteration >= 10)

        src_couples = sorted(src_couples, key=lambda s: len(s[1]))

        train_decoder_length, test_decoder_length = 0.0, 0.0
        # choose items to put into test set
        test_ids = random.sample(range(num_iteration), num_test)
        for i in range(num_iteration):
            batch_seq_couples = sorted(src_couples[i * batch_size: (i + 1) * batch_size], key=lambda s: len(s[0]),
                                       reverse=True)
            decoder_length = float(sum([len(x[1]) for x in batch_seq_couples])) / 64

            input_group, target_group = self.__process(batch_seq_couples)
            if i not in test_ids:
                self.data.append((input_group, target_group))
                train_decoder_length += decoder_length
            else:
                # test data
                self.test.append((input_group, target_group))
                test_decoder_length += decoder_length

        self.train_data_len = len(self.data)
        self.test_data_len = len(self.test)
        mean_train_decoder_len = train_decoder_length / self.train_data_len
        mean_test_decoder_len = test_decoder_length / self.test_data_len
        print('Mean decoder length: (%.2f, %.2f)' % (mean_train_decoder_len, mean_test_decoder_len))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_vocabulary_size(self):
        return self.vocabulary.num_words

    def random_batch(self):
        # assert(self.train_data_len > 1)
        return self.data[random.randint(0, self.train_data_len - 1)]

    def random_test(self):
        # assert(self.test_data_len > 1)
        return self.test[random.randint(0, self.test_data_len - 1)]

    def __process(self, batch_seq_couples):
        input_seqs, target_seqs = tuple(zip(*batch_seq_couples))
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
        input_var = torch.tensor(input_padded, device=DEVICE).transpose(0, 1)
        target_var = torch.tensor(target_padded, device=DEVICE).transpose(0, 1)

        return (input_var, input_lens), (target_var, target_lens)

    def indexes_from_sentence(self, sentence):
        return [self.vocabulary.word2index[word] for word in sentence] + [EOS_token]

    @staticmethod
    def pad_seq(seq, max_length):
        seq += [PAD_token for _ in range(max_length - len(seq))]
        return seq
