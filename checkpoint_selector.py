# coding=utf8
import json
import math

from numpy.ma import arange

from data_utils import build_data_loader
from model_utils import load_vocabulary, build_model, model_evaluate

with open('config.json') as config_file:
    config = json.load(config_file)

MIN_EPOCH = config['SELECTOR']['MIN_EPOCH']
MAX_EPOCH = config['SELECTOR']['MAX_EPOCH']
STEP_SIZE = config['SELECTOR']['STEP_SIZE']
BATCH_SIZE = config['TRAIN']['BATCH_SIZE']


def select_proper_checkpoint():
    min_loss = 99999
    ideal_epoch = 0
    for epoch in arange(MIN_EPOCH, MAX_EPOCH + STEP_SIZE, STEP_SIZE):
        vocab = load_vocabulary()
        model = build_model(len(vocab.word2index), load_checkpoint=True, checkpoint_epoch=epoch, print_module=False)
        data_set = build_data_loader(batch_size=BATCH_SIZE)
        test_loss = model_evaluate(model, data_set)
        print('EPOCH %d Test PPL: %.4f' % (epoch, math.exp(test_loss)))
        if min_loss > test_loss:
            min_loss = test_loss
            ideal_epoch = epoch

    print("Ideal EPOCH: %d, Min PPL %.4f" % (ideal_epoch, math.exp(min_loss)))


if __name__ == '__main__':
    try:
        select_proper_checkpoint()
    except KeyboardInterrupt as _:
        print("You quit.")
