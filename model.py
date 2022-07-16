# coding=utf8
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from custom_token import *

with open('config.json') as config_file:
    config = json.load(config_file)

DEVICE = torch.device(config['TRAIN']['DEVICE'])


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, max_length=20, tie_weights=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length

        if DEVICE != torch.device("cpu"):
            self.encoder.to(device=DEVICE)
            self.decoder.to(device=DEVICE)

        if tie_weights:
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def forward(self, input_group, target_group=(None, None), teacher_forcing_ratio=0.5):
        input_var, input_lens = input_group
        encoder_outputs, encoder_hidden = self.encoder(input_var, input_lens)

        batch_size = input_var.size(1)
        target_var, target_lens = target_group
        if target_var is None or target_lens is None:
            max_target_length = self.max_length
            # without teacher forcing
            teacher_forcing_ratio = 0
        else:
            max_target_length = max(target_lens)

        # store all decoder outputs
        all_decoder_outputs = torch.zeros(max_target_length, batch_size, self.decoder.output_size, device=DEVICE)
        # first decoder input
        decoder_input = torch.tensor([GO_token] * batch_size, requires_grad=False, device=DEVICE)
        decoder_hidden = encoder_hidden[:self.decoder.num_layers]
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = \
                self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_outputs[t] = decoder_output
            # select real target or decoder output
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = target_var[t]
            else:
                # decoder_output = F.log_softmax(decoder_output)
                top_v, top_i = decoder_output.data.topk(1, dim=1)
                decoder_input = top_i.squeeze(1).clone().detach()

        return all_decoder_outputs

    def response(self, input_var):
        # input_var size (length, 1)
        length = input_var.size(0)
        input_group = (input_var, [length])
        # outputs size (max_length, output_size)
        decoder_outputs = self.forward(input_group, teacher_forcing_ratio=0)
        # top_v, top_i = decoder_outputs.data.top_k(1, dim=1)
        # decoder_index = top_i.squeeze(1)
        # return decoder_index
        return decoder_outputs


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(weight_init.xavier_uniform(torch.tensor(1, self.hidden_size, device=DEVICE)),
                                  requires_grad=False)

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.batch_score(hidden, encoder_outputs)
        return func.softmax(attn_energies, dim=1).unsqueeze(1)

    # faster
    def batch_score(self, hidden, encoder_outputs):
        if self.method == 'dot':
            # encoder_outputs size (batch_size, hidden_size, length)
            encoder_outputs = encoder_outputs.permute(1, 2, 0)
            return torch.bmm(hidden.transpose(0, 1), encoder_outputs).squeeze(1)
        elif self.method == 'general':
            length = encoder_outputs.size(0)
            batch_size = encoder_outputs.size(1)
            energy = self.attn(encoder_outputs.view(-1, self.hidden_size)).view(length, batch_size, self.hidden_size)
            return torch.bmm(hidden.transpose(0, 1), energy.permute(1, 2, 0)).squeeze(1)
        elif self.method == 'concat':
            length = encoder_outputs.size(0)
            batch_size = encoder_outputs.size(1)
            attn_input = torch.cat((hidden.repeat(length, 1, 1), encoder_outputs), dim=2)
            energy = self.attn(attn_input.view(-1, 2 * self.hidden_size)).view(length, batch_size, self.hidden_size)
            return torch.bmm(self.v.repeat(batch_size, 1, 1), energy.permute(1, 2, 0)).squeeze(1)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1, bidirectional=True):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional,
                          device=DEVICE)

    def forward(self, inputs_seqs, input_lens, hidden=None):
        # embedded size (max_len, batch_size, hidden_size)
        embedded = self.embedding(inputs_seqs)
        packed = pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(outputs)
        if self.bidirectional:
            # sum bidirectional outputs
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # outputs size (max_len, batch_size, hidden_size)
        # hidden size (bi * num_layers, batch_size, hidden_size)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, attn_method, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout, device=DEVICE)
        self.concat = nn.Linear(hidden_size * 2, hidden_size, device=DEVICE)
        self.out = nn.Linear(hidden_size, output_size, device=DEVICE)

        self.attn = Attn(attn_method, hidden_size)

    def forward(self, input_seqs, last_hidden, encoder_outputs):
        # input_seqs size (batch_size,)
        # last_hidden size (num_layers, batch_size, hidden_size)
        # encoder_outputs size (max_len, batch_size, hidden_size)
        batch_size = input_seqs.size(0)
        # embedded size (1, batch_size, hidden_size)
        embedded = self.embedding(input_seqs).unsqueeze(0)
        # output size (1, batch_size, hidden_size)
        output, hidden = self.gru(embedded, last_hidden)
        # attn_weights size (batch_size, 1, max_len)
        attn_weights = self.attn(output, encoder_outputs)
        # context size (batch_size, 1, hidden_size) = (batch_size, 1, max_len) * (batch_size, max_len, hidden_size)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # concat_input size (batch_size, hidden_size * 2)
        concat_input = torch.cat((output.squeeze(0), context.squeeze(1)), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # output size (batch_size, output_size)
        output = self.out(concat_output)
        return output, hidden, attn_weights
