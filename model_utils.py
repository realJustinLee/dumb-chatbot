# coding=utf8
import os

import data_utils
from custom_token import *
from data_utils import Vocabulary
from masked_cross_entropy import *
from model import Seq2Seq, Encoder, Decoder

with open('config.json') as config_file:
    config = json.load(config_file)

CHECKPOINT_PATH = config['TRAIN']['PATH']
DEVICE = torch.device(config['TRAIN']['DEVICE'])
IMPORT_FROM_CUDA = config['LOADER']['IMPORT_FROM_CUDA']
BATCH_SIZE = config['TRAIN']['BATCH_SIZE']

questions = []
with open('test_questions.txt') as question_file:
    for line in question_file:
        questions.append(line[:-1])


def model_evaluate(model, data_set, evaluate_num=10, auto_test=True):
    model.train(False)
    total_loss = 0.0
    for _ in range(evaluate_num):
        input_group, target_group = data_set.random_test()
        all_decoder_outputs = model(input_group, target_group, teacher_forcing_ratio=1)
        target_var, target_lens = target_group
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),
            target_var.transpose(0, 1).contiguous(),
            target_lens
        )
        total_loss += loss.data
        # format_output(data_set.vocabulary.index2word, input_group, target_group, all_decoder_outputs)
    if auto_test is True:
        bot = BotAgent(model, data_set.vocabulary)
        for question in questions:
            print('> %s' % question)
            print('bot: %s' % bot.response(question))
    model.train(True)
    return total_loss / evaluate_num


def build_model(vocab_size, load_checkpoint=False, checkpoint_epoch=-1, print_module=True):
    hidden_size = config['MODEL']['HIDDEN_SIZE']
    attn_method = config['MODEL']['ATTN_METHOD']
    num_encoder_layers = config['MODEL']['N_ENCODER_LAYERS']
    dropout = config['MODEL']['DROPOUT']
    encoder = Encoder(vocab_size, hidden_size, num_encoder_layers, dropout=dropout)
    decoder = Decoder(hidden_size, vocab_size, attn_method, num_encoder_layers, dropout=dropout)
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        max_length=config['LOADER']['MAX_LENGTH'],
        tie_weights=config['MODEL']['TIE_WEIGHTS'],
    )
    if print_module:
        print(model)
    if load_checkpoint is True and os.path.exists(CHECKPOINT_PATH) is True:
        # load checkpoint
        prefix = config['TRAIN']['PREFIX']
        model_path = None
        if checkpoint_epoch >= 0:
            model_path = '%s%s_%d' % (CHECKPOINT_PATH, prefix, checkpoint_epoch)
        else:
            # use last checkpoint
            checkpoints = []
            for root, dirs, files in os.walk(CHECKPOINT_PATH):
                for f_name in files:
                    f_name_sp = f_name.split('_')
                    if len(f_name_sp) == 2:
                        checkpoints.append(int(f_name_sp[1]))
            if len(checkpoints) > 0:
                model_path = '%s%s_%d' % (CHECKPOINT_PATH, prefix, max(checkpoints))

        if model_path is not None and os.path.exists(model_path):
            if IMPORT_FROM_CUDA:
                loaded = torch.load(model_path, map_location=lambda storage, loc: storage)
            else:
                loaded = torch.load(model_path)

            model.load_state_dict(loaded)
            print('Load %s' % model_path)

    # print('Seq2Seq parameters:')
    # for name, param in model.state_dict().items():
    #     print(name, param.size())
    if DEVICE != "cpu":
        model = model.to(device=DEVICE)
    return model


def init_path():
    if os.path.exists(CHECKPOINT_PATH) is False:
        os.mkdir(CHECKPOINT_PATH)


def save_model(model, epoch):
    init_path()
    save_path = '%s%s_%d' % (CHECKPOINT_PATH, config['TRAIN']['PREFIX'], epoch)
    torch.save(model.state_dict(), save_path)


def save_vocabulary(vocabulary_list):
    init_path()
    with open(CHECKPOINT_PATH + config['TRAIN']['VOCABULARY'], 'w') as file:
        for word, index in vocabulary_list:
            file.write('%s %d\n' % (word, index))


def load_vocabulary():
    if os.path.exists(CHECKPOINT_PATH + config['TRAIN']['VOCABULARY']):
        word2index = {}
        with open(CHECKPOINT_PATH + config['TRAIN']['VOCABULARY']) as file:
            for _line in file:
                line_spl = _line[:-1].split()
                word2index[line_spl[0]] = int(line_spl[1])
        index2word = dict(zip(word2index.values(), word2index.keys()))
        vocab = Vocabulary()
        vocab.word2index = word2index
        vocab.index2word = index2word
        return vocab
    else:
        raise Exception('not found %s' % CHECKPOINT_PATH + config['TRAIN']['VOCABULARY'])


class BotAgent(object):
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    def response(self, question):
        input_var = self.build_input_var(question)
        if input_var is None:
            return "Sorry, I don 't know ."
        decoder_output = self.model.response(input_var)
        decoder_output = decoder_output.squeeze(1)
        top_v, top_i = decoder_output.data.topk(1, dim=1)
        top_i = top_i.squeeze(1)
        if DEVICE != "cpu":
            predict_resp = top_i.cpu().numpy()
        else:
            predict_resp = top_i.numpy()
        resp_words = self.build_sentence(predict_resp)
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
            input_var = torch.tensor([words_index], device=DEVICE).transpose(0, 1)
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
