# coding=utf8
import math
import time

from torch import optim
from torch.nn.utils import clip_grad_norm

from data_utils import build_data_loader
from masked_cross_entropy import *
from model_utils import build_model, save_model, model_evaluate, save_vocabulary

with open('config.json') as config_file:
    config = json.load(config_file)
USE_CUDA = config['TRAIN']['CUDA']

n_epochs = config['TRAIN']['N_EPOCHS']
batch_size = config['TRAIN']['BATCH_SIZE']
clip = config['TRAIN']['CLIP']
learning_rate = config['TRAIN']['LEARNING_RATE']
teacher_forcing_ratio = config['TRAIN']['TEACHER_FORCING_RATIO']

print_every = 200
save_every = print_every * 10


def main():
    data_set = build_data_loader(batch_size=batch_size)
    vocabulary_list = sorted(data_set.vocabulary.word2index.items(), key=lambda x: x[1])
    save_vocabulary(vocabulary_list)
    vocab_size = data_set.get_vocabulary_size()
    model = build_model(vocab_size)
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start = time.time()
    data_set_len = len(data_set)
    epoch = 0
    print_loss_total = 0.0
    print('Start Training.')
    while epoch < n_epochs:
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
        print_loss_total += loss.data[0]
        loss.backward()
        clip_grad_norm(model.parameters(), clip)
        # update parameters
        model_optimizer.step()

        if epoch % print_every == 0:
            test_loss = model_evaluate(model, data_set)
            print_summary(start, epoch, math.exp(print_loss_total / print_every))
            print('Test PPL: %.4f' % math.exp(test_loss))
            print_loss_total = 0.0
            if epoch % save_every == 0:
                save_model(model, epoch)
        # break
    save_model(model, epoch)


def print_summary(start, epoch, print_ppl_avg):
    output_log = '%s (epoch: %d finish: %d%%) PPL: %.4f' % \
                 (time_since(start, float(epoch) / n_epochs), epoch, float(epoch) / n_epochs * 100, print_ppl_avg)
    print(output_log)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


if __name__ == '__main__':
    main()
