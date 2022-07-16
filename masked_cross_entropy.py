# coding=utf8
import json

import torch
from torch.nn import functional

with open('config.json') as config_file:
    config = json.load(config_file)

DEVICE = torch.device(config['TRAIN']['DEVICE'])


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = seq_range_expand.clone().detach()
    seq_range_expand = seq_range_expand.to(device=sequence_length.device)
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return torch.lt(seq_range_expand, seq_length_expand)


def masked_cross_entropy(logic, target, length):
    length = torch.tensor(length, device=DEVICE)

    """
    Args:
        logic: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            un-normalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logic_flat: (batch * max_len, num_classes)
    logic_flat = logic.view(-1, logic.size(-1))
    # log_probability_flat: (batch * max_len, num_classes)
    log_probability_flat = functional.log_softmax(logic_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probability_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss
