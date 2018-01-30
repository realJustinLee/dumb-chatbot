# coding=utf8
import os
import sys
import json
import random
import torch
import data_utils
from model import Seq2Seq, Encoder, Decoder
from data_utils import Vocabulary
from masked_cross_entropy import *
from custom_token import *

with open('config.json') as config_file:
    config = json.load(config_file)

CKPT_PATH = config['TRAIN']['PATH']
USE_CUDA = config['TRAIN']['CUDA']

batch_size = config['TRAIN']['BATCH_SIZE']

question_list = []
with open('test_questions.txt') as file:
    for line in file:
        question_list.append(line[:-1])


def model_evaluate(model, dataset, evaluate_num=10, auto_test=True):
    model.train(False)
    total_loss = 0.0
    for _ in range(evaluate_num):
        input_group, target_group = dataset.random_test()
        all_decoder_outputs = model(input_group, target_group, teacher_forcing_ratio=1)
        target_var, target_lens = target_group
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),
            target_var.transpose(0, 1).contiguous(),
            target_lens
        )
        total_loss += loss.data[0]
        # format_output(dataset.vocabulary.index2word, input_group, target_group, all_decoder_outputs)
    if auto_test is True:
        bot = BotAgent(model, dataset.vocabulary)
        for question in question_list:
            print('> %s' % question)
            print('bot: %s' % bot.response(question))
    model.train(True)
    return total_loss / evaluate_num


def build_model(vocab_size, load_ckpt=False, ckpt_epoch=-1):
    hidden_size = config['MODEL']['HIDDEN_SIZE']
    attn_method = config['MODEL']['ATTN_METHOD']
    n_encoder_layers = config['MODEL']['N_ENCODER_LAYERS']
    dropout = config['MODEL']['DROPOUT']
    encoder = Encoder(vocab_size, hidden_size, n_encoder_layers, dropout=dropout)
    decoder = Decoder(hidden_size, vocab_size, attn_method, dropout=dropout)
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        max_length=config['LOADER']['MAX_LENGTH'],
        tie_weights=config['MODEL']['TIE_WEIGHTS']
    )
    print(model)
    if load_ckpt is True and os.path.exists(CKPT_PATH) is True:
        # load checkpoint
        prefix = config['TRAIN']['PREFIX']
        model_path = None
        if ckpt_epoch >= 0:
            model_path = '%s%s_%d' % (CKPT_PATH, prefix, ckpt_epoch)
        else:
            # use last checkpoint
            ckpts = []
            for root, dirs, files in os.walk(CKPT_PATH):
                for fname in files:
                    fname_sp = fname.split('_')
                    if len(fname_sp) == 2:
                        ckpts.append(int(fname_sp[1]))
            if len(ckpts) > 0:
                model_path = '%s%s_%d' % (CKPT_PATH, prefix, max(ckpts))

        if model_path is not None and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print('Load %s' % model_path)

    # print('Seq2Seq parameters:')
    # for name, param in model.state_dict().items():
    #     print(name, param.size())
    if USE_CUDA:
        model = model.cuda()
    return model


def init_path():
    if os.path.exists(CKPT_PATH) is False:
        os.mkdir(CKPT_PATH)


def save_model(model, epoch):
    init_path()
    save_path = '%s%s_%d' % (CKPT_PATH, config['TRAIN']['PREFIX'], epoch)
    torch.save(model.state_dict(), save_path)


def save_vocabulary(vocabulary_list):
    init_path()
    with open(CKPT_PATH + config['TRAIN']['VOCABULARY'], 'w') as file:
        for word, index in vocabulary_list:
            file.write('%s %d\n' % (word, index))


def load_vocabulary():
    if os.path.exists(CKPT_PATH + config['TRAIN']['VOCABULARY']):
        word2index = {}
        with open(CKPT_PATH + config['TRAIN']['VOCABULARY']) as file:
            for line in file:
                line_spl = line[:-1].split()
                word2index[line_spl[0]] = int(line_spl[1])
        index2word = dict(zip(word2index.values(), word2index.keys()))
        vocab = Vocabulary()
        vocab.word2index = word2index
        vocab.index2word = index2word
        return vocab
    else:
        raise ('not found %s' % CKPT_PATH + config['TRAIN']['VOCABULARY'])


class BotAgent(object):
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    def response(self, question):
        input_var = self.build_input_var(question)
        if input_var is None:
            return "sorry, I don 't know ."
        decoder_output = self.model.response(input_var)
        decoder_output = decoder_output.squeeze(1)
        topv, topi = decoder_output.data.topk(1, dim=1)
        topi = topi.squeeze(1)
        if USE_CUDA:
            preidct_resp = topi.cpu().numpy()
        else:
            preidct_resp = topi.numpy()
        resp_words = self.build_sentence(preidct_resp)
        return resp_words

    def build_input_var(self, user_input):
        words = data_utils.basic_tokenizer(user_input)
        words_index = []
        unknown_words = []
        for word in words:
            if word in self.vocab.word2index.keys():
                # keep known words
                words_index.append(self.vocab.word2index[word])
            else:
                unknown_words.append(word)
        if len(unknown_words) > 0:
            print('unknown_words: ' + str(unknown_words))
        # append EOS token
        words_index.append(EOS_token)
        if len(words_index) > 0:
            input_var = Variable(torch.LongTensor([words_index])).transpose(0, 1)
            if USE_CUDA:
                input_var = input_var.cuda()
            # input_var size (length, 1)
            return input_var
        return None

    def build_sentence(self, words_index):
        resp_words = []
        for index in words_index:
            if index < 3:
                # end sentence
                break
            resp_words.append(self.vocab.index2word[index])
        return ' '.join(resp_words)


if __name__ == '__main__':
    pass
