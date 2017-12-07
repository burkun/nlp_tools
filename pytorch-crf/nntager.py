#! /usr/bin/env python
# -*- coding: gb18030 -*-
# vim:fenc=gb18030

"""
crf++ impletemention
"""
from nnlayer import CrfLayer
from nnlayer import WideEmbedding
import torch
import torch.nn as nn


#last, trans(S * B), fill_val
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


class PureCrf(nn.Module):
    def __init__(self, feature_num, tag_size):
        super(PureCrf, self).__init__()
        self.crf = CrfLayer(tag_size)
        self.tag_size = tag_size
        self.feature_embed = WideEmbedding(feature_num)
        self.added = torch.autograd.Variable(torch.LongTensor(range(0, tag_size)))

    def forward(self, input_words, input_length, is_viterbi=False):
        # x shape = S * B * Tag * template_feature_size
        step_num = input_words.size(0)
        feats = self._get_emit_feature(input_words, input_length)
        # get mask
        mask = make_mask(input_length, step_num, True)
        if feats.is_cuda:
            mask = mask.cuda(feats.get_device())
        forward_score, max_s, max_s_idx = self.crf(feats, mask, is_viterbi)
        return forward_score, max_s, max_s_idx, feats

    def get_loss(self, input_words, input_tags, input_length):
        # x shape = S * B * 1 * template_feature_size
        step_num = input_words.size(0)
        feats = self._get_emit_feature(input_words, input_length)
        # get mask
        mask = make_mask(input_length, step_num, True)
        if feats.is_cuda:
            mask = mask.cuda(feats.get_device())
        Z, _, _ = self.crf(feats, mask, False)
        path_mask = make_mask(input_length, feats.size(0),  is_only_last=False, trans_pos = True, fill_val = False)
        if feats.is_cuda:
            path_mask = path_mask.cuda(feats.get_device())
        gold_score = self.crf.score_path(feats, input_tags, path_mask)
        return Z - gold_score

    def get_viterbi_path(self, input_words, input_length):
        _, max_s, max_s_idx, _ = self.forward(input_words, input_length, is_viterbi=True)
        return self.crf.viterbi(max_s, max_s_idx, input_length)


    def _get_emit_feature(self, input_words, input_length):
        # input_words shape = S * B * template_feature_size
        # extend to full tag
        step_num = input_words.size(0)
        batch_num = input_words.size(1)
        feature_num = input_words.size(2)

        # S * B
        # TRUE - > TRUE -> FALSE
        mask = make_mask(input_length, step_num, False, True, False)
        added = self.added.view(1, 1, self.tag_size, 1)
        if input_words.is_cuda:
            mask = mask.cuda(input_words.get_device())
            added = added.cuda(input_words.get_device())

        added = added.expand(step_num, batch_num, self.tag_size, feature_num)
        x = input_words.view(step_num, batch_num, 1, feature_num)
        x = x.expand(step_num, batch_num, self.tag_size, feature_num)
        x = x + added

        mask = mask.view(step_num, batch_num, 1, 1).expand(step_num, batch_num, self.tag_size, feature_num)
        x.masked_fill_(mask, 0)

        x = self.feature_embed(x)
        # x shape = S * B * Tag * feature_weight
        x = x.sum(-1)
        # x shape = S * B * T
        feats = x.view(step_num, batch_num, -1)
        return feats

