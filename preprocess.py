# coding=utf8
import random
import json
import re
import data_utils

with open('config.json') as config_file:
    config = json.load(config_file)

DATA_PATH = config['DATA']['PATH']
MOVIE_CONVERSATIONS = config['DATA']['MOVIE_CONVERSATIONS']
MOVIE_LINES = config['DATA']['MOVIE_LINES']
DIALOGUE_CORPUS = config['DATA']['DIALOGUE_CORPUS']


def load_conversations(file_path):
    dialogue_list = []
    with open(file_path) as file:
        for line in file:
            sp = line[:-1].split(' +++$+++ ')
            ua, ub, m = sp[:3]
            line_list = eval(sp[3])
            dialogue_list.append(line_list)
    return dialogue_list


def load_movie_lines(file_path):
    id2sentence = {}
    with open(file_path) as file:
        for line in file:
            sp = line[:-1].split(' +++$+++ ')
            lid, sentence = sp[0], sp[4]
            id2sentence[lid] = sentence
    return id2sentence


def export_dialogue_corpus():
    dialogues = load_conversations(DATA_PATH + MOVIE_CONVERSATIONS)
    id2sentence = load_movie_lines(DATA_PATH + MOVIE_LINES)
    questions, answers = [], []
    for ids in dialogues:
        length = len(ids) if len(ids) % 2 == 0 else len(ids) - 1
        for i in range(length):
            sentence = ' '.join(data_utils.basic_tokenizer(id2sentence[ids[i]]))
            if i % 2 == 0:
                questions.append(sentence)
            else:
                answers.append(sentence)
    dialogue_groups = zip(questions, answers)
    print('Dialogue pairs: %d' % len(dialogue_groups))

    # random.shuffle(dialogue_corpus)
    with open(DATA_PATH + DIALOGUE_CORPUS, 'w') as file:
        for a, b in dialogue_groups:
            file.write('%s +++$+++ %s\n' % (a, b))


if __name__ == '__main__':
    export_dialogue_corpus()
