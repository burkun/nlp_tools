#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 漏 2018 booker <oognit@live.com>
#
# Distributed under terms of the MIT license.

"""
seq2seq langugage model
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

def make_mask(sent_len, step_num,  is_only_last=True, trans_pos = True, fill_val = True):
    # batch * 1
    batch_size = len(sent_len)
    mask = torch.ByteTensor(step_num, batch_size).fill_(not fill_val)
    if is_only_last:
        for batch_idx in range(batch_size):
            mask[sent_len[batch_idx] - 1, batch_idx] = fill_val
    else:
        for batch_idx in range(batch_size):
            mask[0:sent_len[batch_idx], batch_idx] = fill_val
    mask = torch.autograd.Variable(mask)
    if not trans_pos:
        mask = mask.t()
        mask.contiguous()
    return mask

class LM(nn.Module):
    def __init__(self, config):
        self._assert_args(config)
        self.voc_size = config["voc_size"]
        self.emb_dim = config["emb_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.dropout = config.get("dropout", 0.5)
        self.pad_val = conig.get("pad_val", 0)
        self._build()

    def _build(self):
        self.embedding = nn.Embedding(self.voc_size, self.emb_dim, self.pad_val)
        self.encoder = nn.LSTM(self.emb_dim, self.hidden_dim, bidirectional = True, dropout=self.dropout)
        # this fc is too large
        self.shared_fc = nn.Linear(self.hidden_dim, self.voc_size)

    def forward(self, input_src, input_src_len):
        batch_size = input_src.size(1)
        sent_len = input_src.size(0)
        embed_src = self.embedding(input_src)
        embeds_packed = nn.utils.rann.pack_padded_sequence(embed_src, input_src_len.numpy(), batch_first = False)
        encoder_out, _ = self.encoder(embeds_packed)
        encoder_out, _ = nn.utils.rnn.pad_packed_sequence(encoder_out, batch_first = False)

        #extract, encoder out size sent_len * batch_size * (hidden_size * num_directions)
        assert encoder_out.size(2) == 2
        # size sent_len * batch_size * hidden_size
        # org sent = <s> A B C D </s>
        # left = A B C D </s>
        # right = D C B A <s>
        left_out = encoder_out[1:, :, :self.hidden_dim]
        left_size = left_out.size()
        right_out = encoder_out[1:, :, self.hidden_dim:]
        right_size = right_out.size()
        left_out = self.shared_fc(-1, left_size[-1]).view(left_size[0], left_size[1], -1)
        right_out = self.shared_fc(-1, right_size[-1]).view(right_size[0], right_size[1], -1)
        left_out = F.softmax(left_out, dim = 2)
        right_out = F.softmax(right_out, dim = 2)
        return left_out, right_out

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.encoder.bias.data.fill(0)
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def _assert_args(self, config):
        assert hasattr(config, "voc_size")
        assert hasattr(config, "emb_dim"))
        assert hasattr(config, "hidden_dim")

    def loss(self, prob_out, true_out, ignore_idx = 0):
        prob_out = prob_out.view(-1, self.voc_size)
        true_out = true_out.view(-1)
        assert prob_out.size(1) == true_out.size(0)
        loss = F.cross_entropy(prob_out, true_out, ignore_idx = ignore_idx)
        return loss



