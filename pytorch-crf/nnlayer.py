#! /usr/bin/env python
# -*- coding: gb18030 -*-
# vim:fenc=gb18030

"""
wideÐÎÊ½µÄfeature
"""
import torch
import torch.nn as nn
import torch.autograd as autograd

class CrfLayer(nn.Module):
    def __init__(self, klass_num):
        super(CrfLayer, self).__init__()
        self.transitions = torch.nn.Parameter(torch.randn(klass_num, klass_num))
        self.alpha_0 = torch.nn.Parameter(torch.randn(klass_num, 1))
        self.klass_num = klass_num

    def forward(self, emits, mask, is_viterbi=False):
        # emits's shape = S * B * tag_size
        # sent_len's shape = B * 1, normal list
        # extend transitions to B * T * T
        # mark last word = 1 in mask
        batch_size = emits.size(1)
        step_num = emits.size(0)
        tags_size = emits.size(2)
        # extend mask
        mask = mask.view(step_num, batch_size, 1).expand(step_num, batch_size, tags_size)
        # batch trans, batch_size * T * T
        batch_trans = torch.cat([self.transitions for _ in range(batch_size)], 0)
        batch_trans = batch_trans.view(batch_size, tags_size, tags_size)

        # Iterate through the sentence, The First alpha
        forward_var = torch.cat([self.alpha_0 for _ in range(batch_size)], 0).view(batch_size, tags_size, 1)
        forward_var = forward_var + emits[0, :, :].view(batch_size, tags_size, 1)

        alphas = [forward_var]
        max_scores = []
        max_scores_pre = []
        for t in xrange(1, step_num):
            forward_var = forward_var.view(batch_size, tags_size, 1).expand(batch_size, tags_size, tags_size)
            # expand is used for replace loop
            current = emits[t, :, :].view(batch_size, 1, tags_size)
            alpha_t = forward_var + current.expand(batch_size, tags_size, tags_size) + batch_trans
            if is_viterbi:
                # cur_max_idx tag posÖÐµÄÖµ±íÊ¾µ±Ç°Î»ÖÃµÄµÃ·Ö×î´óµÄÊÇ´ÓÇ°Ò»¸öµÄÄÇ¸öµØ·½À´µÄ
                # eg forward = [1, 2], trans = [[2,3],[4,5]] emit = [10, 1]
                # trans[pre][cur]
                #    1 + 2 + 10     |    1 + 3 + 1     (0->0) (0->1)
                #    2 + 4 + 10     |    2 + 5 + 1     (1->0) (1->1)
                # ¶ÔµÚ0Î¬maxÖ®ºó£¬±íÊ¾µÄÊÇ´Ópre->curµÄ×î´óµÄÄÇ¸ös
                cur_max_score, cur_max_idx = torch.max(alpha_t, 1)
                max_scores.append(cur_max_score)
                max_scores_pre.append(cur_max_idx)
            log_alpha_t = CrfLayer.log_sum_exp(alpha_t, 1).view(batch_size, tags_size, 1)
            forward_var = log_alpha_t
            alphas.append(log_alpha_t)

        alphas = torch.cat(alphas, 0).view(step_num, batch_size, tags_size)
        # step_num, batch_size, tags_size
        # 1 * batch_size * tags_size
        last_alphas = torch.masked_select(alphas, mask).view(batch_size, tags_size, 1)
        # alpha_z = sum
        alpha_z = torch.sum(CrfLayer.log_sum_exp(last_alphas, axis=1))
        if is_viterbi:
            max_scores = torch.cat(max_scores, 0).view(step_num - 1, batch_size, tags_size)
            max_scores_pre = torch.cat(max_scores_pre, 0).view(step_num - 1, batch_size, tags_size)
            return alpha_z, max_scores, max_scores_pre
        else:
            return alpha_z, None, None

    @staticmethod
    def log_sum_exp(x, axis=None):
        #shape = 1 * batch * tags
        x_max, _ = torch.max(x, axis)
        x_max_expand = x_max.expand(x.size(0), x.size(1), x.size(2))
        return x_max + torch.log(torch.sum(torch.exp(x - x_max_expand), axis))

    def viterbi(self, max_scores, max_scores_pre, sent_length):
        # TODO speed up
        # sent_length = B * 1
        # max_scores = (S-1) * B * T
        # max_scores_pre = (S-1) * B * T
        best_paths = []
        batch_size = max_scores.size(1)
        for m in range(batch_size):
            cur_path = []
            _, last_max_node = torch.max(max_scores[sent_length[m] - 2][m], 0)
            last_max_node = last_max_node.data[0]
            cur_path.append(last_max_node)
            # last_max_node = 0
            for t in range(sent_length[m] - 2, -1, -1):
                last_max_node = max_scores_pre[t][m][last_max_node].data[0]
                cur_path.append(last_max_node)
            cur_path = cur_path[::-1]
            best_paths.append(cur_path)
        return best_paths

    def score_path(self, emits, tags, mask = None):
        # tags's shape = S * B
        # sent_length = B * 1
        # emits = S * B * tags_size
        # tags's idx range = [0, tags_size)
        # create tag trans matrix

        batch_size = emits.size(1)
        step_num = emits.size(0)

        #trans score
        tags_index = tags * self.klass_num
        tags_index = torch.cat([torch.autograd.Variable(tags.data.new(1, batch_size).fill_(0)), tags_index[:step_num-1]], 0)
        tags_index = (tags_index + tags)
        trans_scores = self.transitions.view(-1).index_select(0, tags_index.view(-1)).view(step_num, batch_size)
        trans_scores[0] = self.alpha_0.view(-1).index_select(0, tags_index[0].view(-1))

        #emit score
        emit_index = torch.LongTensor(range(0, batch_size * step_num)) * self.klass_num
        emit_index = torch.autograd.Variable(emit_index)
        if emits.is_cuda:
            emit_index = emit_index.cuda(emits.get_device())
        tags = tags.contiguous()
        emit_tags = emit_index.view(-1).contiguous() + tags.view(-1).contiguous()
        emit_scores = emits.view(-1).index_select(0, emit_tags).view(step_num, batch_size)
        total_scores = emit_scores + trans_scores
        if mask is not None:
            total_scores.masked_fill_(mask, 0)
        # total score, batched score

        return total_scores.sum()

'''
µ×²ãµÄfeature´æ´¢
'''
class WideEmbedding(nn.Module):
    def __init__(self, feature_num):
        # feature num not include unk
        super(WideEmbedding, self).__init__()
        self.feature_num = feature_num
        self.weights = nn.Parameter(torch.cat([torch.FloatTensor([0]),torch.randn(self.feature_num)], dim = 0))

    def forward(self, x):
        # x shape = S * BATCH * TAG_SIZE * TEMPLATE_SIZE
        size = x.size()
        x = x.view(-1)
        x = torch.index_select(self.weights, 0, x)
        x = x.view(size)
        return x

    def forzen(self):
        for param in self.parameters():
            param.requires_grad = False

if __name__ == "__main__":
    pass
    #import time
    #a = time.time()
    #torch.manual_seed(0)
    #cl = CrfLayer(5)
    #emits = autograd.Variable(torch.randn(100, 64, 5))
    #tags = autograd.Variable(torch.LongTensor([1]).view(1, 1, 1).expand(100, 64, 1))
    #print cl.score_path(emits, tags)
    #print time.time() - a
