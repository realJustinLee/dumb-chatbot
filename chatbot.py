# coding=utf8
import json
import os
import platform
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from model_utils import load_vocabulary, build_model, BotAgent

with open('config.json') as config_file:
    config = json.load(config_file)

BOT_NAME = config['TEST']['BOT_NAME']
ckpt_epoch = config['TEST']['CKPT_EPOCH']


def main():
    vocab = load_vocabulary()
    model = build_model(len(vocab.word2index), load_ckpt=True, ckpt_epoch=ckpt_epoch)
    bot = BotAgent(model, vocab)
    while True:
        user_input = raw_input('me: ')
        if user_input.strip() == '':
            continue
        response = bot.response(user_input)
        print('%s: %s' % (BOT_NAME, response))
        curr_sys = platform.system()
        if curr_sys == 'Linux':
            os.system('echo %s | festival --tts' % response)
        elif curr_sys == 'Darwin':
            os.system('say &s' % response)


if __name__ == '__main__':
    main()
