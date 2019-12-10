# coding=utf8
import json
import os

import data_utils

with open('config.json') as config_file:
    config = json.load(config_file)

DATA_PATH = config['DATA']['PATH']
MOVIE_CONVERSATIONS = config['DATA']['MOVIE_CONVERSATIONS']
MOVIE_LINES = config['DATA']['MOVIE_LINES']
DIALOGUE_CORPUS = config['DATA']['DIALOGUE_CORPUS']


def load_conversations(file_path):
    dialogue_ids_list = []
    with open(file_path, errors='ignore') as conversation_file:
        for row in conversation_file:
            split_line = row[:-1].split(' +++$+++ ')
            ua, ub, m = split_line[:3]
            dialogue_ids = eval(split_line[3])
            dialogue_ids_list.append(dialogue_ids)
    return dialogue_ids_list


def load_movie_lines(file_path):
    id2sentence = {}
    with open(file_path, errors='ignore') as movie_line_file:
        for row in movie_line_file:
            split_line = row[:-1].split(' +++$+++ ')
            dialogue_id, sentence = split_line[0], split_line[4]
            id2sentence[dialogue_id] = sentence
    return id2sentence


def export_dialogue_corpus():
    dialogue_ids_list = load_conversations(os.path.join(os.path.abspath(DATA_PATH), MOVIE_CONVERSATIONS))
    id2sentence = load_movie_lines(os.path.join(os.path.abspath(DATA_PATH), MOVIE_LINES))
    questions, answers = [], []
    for ids in dialogue_ids_list:
        length = len(ids) if len(ids) % 2 == 0 else len(ids) - 1
        for i in range(length):
            sentence = ' '.join(data_utils.basic_tokenizer(id2sentence[ids[i]]))
            if i % 2 == 0:
                questions.append(sentence)
            else:
                answers.append(sentence)
    dialogue_couples = list(zip(questions, answers))
    print('Dialogue couples: %d' % len(dialogue_couples))

    # random.shuffle(dialogue_corpus)
    with open(os.path.join(os.path.abspath(DATA_PATH), DIALOGUE_CORPUS), 'w') as dialogue_file:
        for question, answer in dialogue_couples:
            dialogue_file.write('%s +++$+++ %s\n' % (question, answer))


if __name__ == '__main__':
    export_dialogue_corpus()
