# coding=utf8
import math
import time

from torch import optim
from torch.nn.utils import clip_grad_norm_

from data_utils import build_data_loader
from masked_cross_entropy import *
from model_utils import build_model, save_model, model_evaluate, save_vocabulary

with open('config.json') as config_file:
    config = json.load(config_file)
USE_CUDA = config['TRAIN']['CUDA']

N_EPOCHS = config['TRAIN']['N_EPOCHS']
BATCH_SIZE = config['TRAIN']['BATCH_SIZE']
CLIP = config['TRAIN']['CLIP']
LEARNING_RATE = config['TRAIN']['LEARNING_RATE']
TEACHER_FORCING_RATIO = config['TRAIN']['TEACHER_FORCING_RATIO']

PRINT_EVERY = 200
SAVE_EVERY = PRINT_EVERY * 10


def train():
    data_set = build_data_loader(batch_size=BATCH_SIZE)
    vocabulary_list = sorted(data_set.vocabulary.word2index.items(), key=lambda x: x[1])
    save_vocabulary(vocabulary_list)
    vocab_size = data_set.get_vocabulary_size()
    model = build_model(vocab_size)
    model_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start = time.time()
    data_set_len = len(data_set)
    epoch = 0
    print_loss_total = 0.0
    print('Start Training.')
    while epoch < N_EPOCHS:
        epoch += 1
        input_group, target_group = data_set.random_batch()
        # zero gradients
        model_optimizer.zero_grad()
        # run seq2seq
        all_decoder_outputs = model(input_group, target_group, teacher_forcing_ratio=1)
        target_var, target_lens = target_group
        # loss calculation and back-propagation
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),
            target_var.transpose(0, 1).contiguous(),
            target_lens
        )
        print_loss_total += loss.data
        loss.backward()
        clip_grad_norm_(model.parameters(), CLIP)
        # update parameters
        model_optimizer.step()

        if epoch % PRINT_EVERY == 0:
            test_loss = model_evaluate(model, data_set)
            print_summary(start, epoch, math.exp(print_loss_total / PRINT_EVERY))
            print('Test PPL: %.4f' % math.exp(test_loss))
            print_loss_total = 0.0
            if epoch % SAVE_EVERY == 0:
                save_model(model, epoch)
        # break
    save_model(model, epoch)


def print_summary(start, epoch, print_ppl_avg):
    output_log = '%s (epoch: %d finish: %d%%) PPL: %.4f' % \
                 (time_since(start, float(epoch) / N_EPOCHS), epoch, float(epoch) / N_EPOCHS * 100, print_ppl_avg)
    print(output_log)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (Time Remaining: %s)' % (as_minutes(s), as_minutes(rs))


if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt as _:
        print("You quit.")
