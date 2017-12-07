#! /usr/bin/env python
# -*- coding: gb18030 -*-
# vim:fenc=gb18030

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

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

class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, vocab, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = vocab['<pad>']
        self.bos = vocab['<s>']
        self.eos = vocab['</s>']
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

        # The attentions (matrix) for each time.
        self.attn = []

        self.lam = 0.1

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]


    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]


        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)


        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True

        return self.done

    def sort_best(self):
        """Sort the beam."""
        #calc sent_len
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1] 





class SoftDotAttention(nn.Module):
    """
    REF: http://www.aclweb.org/anthology/D15-1166
    """

    def __init__(self, target_dim, source_dim):
        super(SoftDotAttention, self).__init__()
        self.fc1 = nn.Linear(target_dim, source_dim, bias=False)
        self.fc2 = nn.Linear(source_dim + target_dim, target_dim, bias=False)

    def forward(self, target_input, ctx, mask = None):
        """
        target_input: batch * dim * 1
        ctx : batch * sourceL * dim
        """
        # batch * s_dim * 1

        target = self.fc1(target_input.view(target_input.size(0), -1))
        target = target.unsqueeze(2)
        # batch (sl * dim) * batch(s_dim * 1) = batch(s1*1) = batch * 1 * s1
        attn = F.softmax(torch.bmm(ctx, target).squeeze(2))
        #attn.data.fill_(0)

        attn3 = attn.view(ctx.size(0), 1, ctx.size(1))
        # batch * 1 * s_dim
        h_tilde = torch.cat([torch.bmm(attn3, ctx).squeeze(1), target_input], 1)
        # concat

        h_tilde = F.tanh(self.fc2(h_tilde))
        return h_tilde, attn

class LocalAttention(nn.Module):
    """
    REF: http://www.aclweb.org/anthology/D15-1166
    """

    def __init__(self, target_dim, source_dim, window_size):
        super(LocalAttention, self).__init__()
        self.fc1 = nn.Linear(target_dim, source_dim, bias=False)
        self.fc2 = nn.Linear(source_dim + target_dim, target_dim, bias=False)
        #for predict position
        self.fc3 = nn.Linear(target_dim, 1, bias=False)
        self.window_size = window_size

    def forward(self, target_input, ctx, ctx_length):
        """
        target_input: batch * dim * 1
        ctx : batch * sourceL * dim
        """
        batch_size = target_input.size(0)
        src_sent_len = ctx.size(1)
        gpu_idx = -1
        if target_input.is_cuda:
            gpu_idx = target_input.get_device()

        #predict
        pt = F.tanh(self.fc3(target_input.view(batch_size, -1)))
        # B * 1
        pt = F.sigmoid(pt).view(-1)

        #init ctx_length B * 1
        ctx_len = torch.autograd.Variable(ctx_length).view(-1)

        input_xpos = torch.autograd.Variable(torch.LongTensor(range(0, src_sent_len)))
        # mask shape b * s
        mask = make_mask(ctx_length, src_sent_len, is_only_last=False, trans_pos = False, fill_val = False)
        if gpu_idx >=0:
            ctx_len = ctx_len.cuda(gpu_idx)
            input_xpos = input_xpos.cuda(gpu_idx)
            mask = mask.cuda(gpu_idx)
            pt = pt.cuda(gpu_idx)
        pt = ctx_len.float() * pt
        pt_ex = pt.view(batch_size, 1).expand(batch_size, src_sent_len)
        # create normal input
        input_xpos = input_xpos.view(1, src_sent_len).expand(batch_size, src_sent_len)
        input_xpos = input_xpos.float() - pt_ex
        att_weigth = self.normal_dis(input_xpos.view(-1)).view(batch_size, src_sent_len)
        # b * s
        #att_weight = att_weigth.masked_fill(mask, 0)


        target = self.fc1(target_input.view(target_input.size(0), -1))
        target = target.unsqueeze(2)
        # batch (sl * dim) * batch(s_dim * 1) = batch(s1*1) = batch * 1 * s1
        attn = F.softmax(torch.bmm(ctx, target).squeeze(2))
        attn = attn * att_weigth
        #attn.data.fill_(0)

        attn3 = attn.view(ctx.size(0), 1, ctx.size(1))
        # batch * 1 * s_dim
        h_tilde = torch.cat([torch.bmm(attn3, ctx).squeeze(1), target_input], 1)
        # concat

        h_tilde = F.tanh(self.fc2(h_tilde))
        return h_tilde, attn

    def normal_dis(self, x):
        return torch.exp(-((8*x/self.window_size)**2))

class LSTMAttentionDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_attn = 1, batch_first=True, source_dim = None):
        super(LSTMAttentionDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(self.input_dim, self.hidden_dim);
        self.batch_first = batch_first
        if source_dim is None:
            source_dim = hidden_dim
        self.source_dim = source_dim
        self.use_attn = use_attn
        self.attn = SoftDotAttention(self.hidden_dim, self.source_dim)
        if use_attn  == 2:
            self.attn = LocalAttention(self.hidden_dim, self.source_dim, 6)


    def forward(self, inputs, hidden, ctx, src_len = None):
        # hidden = h0, c0
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
            # ctx = ctx.transpose(0, 1)
        # s * b * d
        steps = inputs.size(0)
        output = []

        for i in range(steps):
            hidden = self.lstm_cell(inputs[i], hidden)
            if self.use_attn > 0:
                hidden_t, attn = self.attn(hidden[0], ctx, src_len)
                output.append(hidden_t)
                hidden = (hidden_t, hidden[1], src_len)
            else:
                output.append(hidden[0])
        output = torch.cat(output, 0).view(inputs.size(0), inputs.size(1), -1)

        if self.batch_first:
            output = output.transpose(0, 1)
        return output, hidden

    def step(self, sing_step, hidden, ctx, src_len = None):
        # B * 1 * DIM
        if self.batch_first:
            sing_step = sing_step.transpose(0, 1)
            # ctx = ctx.transpose(0, 1)
        # s * b * d
        hidden = self.lstm_cell(sing_step[0], hidden)
        if self.use_attn > 0:
            hidden_t, _ = self.attn(hidden[0], ctx, src_len)
            hidden = (hidden_t, hidden[1])
        return hidden


class Seq2Seq(nn.Module):
    def __init__(self, voc_size, emb_dim, src_hidden_dim, trg_hidden_dim, dropout = 0, pad_val =0, batch_first = True, use_attn = 1):
        super(Seq2Seq, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.dropout = dropout
        self.pad_val = pad_val
        #encoder layer
        self.encoder_layers = 1
        self.batch_first = batch_first

        self.embedding = nn.Embedding(voc_size, emb_dim, self.pad_val)
        self.encoder = nn.LSTM(self.emb_dim, self.src_hidden_dim // 2, self.encoder_layers, bidirectional = True, batch_first = self.batch_first, dropout = self.dropout)
        self.decoder = LSTMAttentionDecoder(self.emb_dim, self.trg_hidden_dim, use_attn = use_attn, batch_first=True, source_dim = self.src_hidden_dim)
        self.e2d = nn.Linear(self.src_hidden_dim, trg_hidden_dim)

        self.decoder2vocab = nn.Linear(trg_hidden_dim, self.voc_size)

    def forward(self, input_src, input_trg, input_src_len):
        # B * S * D
        batch_size = input_src.size(0)

        embed_src = self.embedding(input_src)
        embed_trg = self.embedding(input_trg)


        embeds_packed = nn.utils.rnn.pack_padded_sequence(embed_src, input_src_len.numpy(), batch_first = self.batch_first)
        encoder_out, hidden = self.encoder(embeds_packed)
        encoder_out, _ = nn.utils.rnn.pad_packed_sequence(encoder_out, batch_first = self.batch_first)

        # batch * s * d


        ht = torch.cat((hidden[0][-1], hidden[0][-2]), 1)
        ct = torch.cat((hidden[1][-1], hidden[1][-2]), 1)

        decoder_h0 = F.tanh(self.e2d(ht))
        decoder_c0 = F.tanh(self.e2d(ct))

        decoder_h0.view(batch_size, 1, self.trg_hidden_dim).unsqueeze(1)
        decoder_c0.view(batch_size, 1, self.trg_hidden_dim).unsqueeze(1)
        # ctx
        decoder_out, hidden = self.decoder(embed_trg, (decoder_h0, decoder_c0), encoder_out, input_src_len)

        # contiguous
        decoder_out = decoder_out.contiguous()
        decoder_logit = self.decoder2vocab(decoder_out.view(-1, self.trg_hidden_dim))
        decoder_logit = decoder_logit.view(decoder_out.size(0), decoder_out.size(1), -1)
        return decoder_logit

    def decode(self, logits):
        size = logits.size()
        logits = logits.view(-1, self.voc_size)
        word_prob = F.softmax(logits)
        word_prob = word_prob.view(size)
        return word_prob

    def predict(self, input_src, input_src_len, word_map, max_len = 20, beam_size = 5):

        # B * S * D
        batch_size = input_src.size(0)

        embed_src = self.embedding(input_src)

        embeds_packed = nn.utils.rnn.pack_padded_sequence(embed_src, input_src_len.numpy(), batch_first = self.batch_first)
        encoder_out, hidden = self.encoder(embeds_packed)
        gpu_idx = -1
        if embed_src.is_cuda:
            gpu_idx = embed_src.get_device()
        #ctx shape b * s * d
        encoder_out, _ = nn.utils.rnn.pad_packed_sequence(encoder_out, batch_first = self.batch_first)

        # batch * s * d


        ht = torch.cat((hidden[0][-1], hidden[0][-2]), 1)
        ct = torch.cat((hidden[1][-1], hidden[1][-2]), 1)

        decoder_h0 = F.tanh(self.e2d(ht))
        decoder_c0 = F.tanh(self.e2d(ct))

        decoder_h0.view(batch_size, 1, self.trg_hidden_dim).unsqueeze(1)
        decoder_c0.view(batch_size, 1, self.trg_hidden_dim).unsqueeze(1)
        # ctx
        #init beign pos
        decoder_h0 = decoder_h0.repeat(beam_size, 1)
        decoder_c0 = decoder_c0.repeat(beam_size, 1)
        hidden = (decoder_h0, decoder_c0)

        beam = [Beam(beam_size, word_map, cuda=True) for k in range(batch_size)]
        # beam * batch, s, dim
        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        input_src_len = input_src_len.repeat(beam_size)
        batch_idx = list(range(batch_size))
        remaining_sents = batch_size

        # rev_word_map
        #rev_word_map = {}
        #for key in word_map:
        #    rev_word_map[word_map[key]] = key
        for idx in range(max_len):
            # get_current_state size is beam size
            target_input = torch.stack([b.get_current_state() for b in beam if not b.done])
            #for words in target_input:
            #    predict = [rev_word_map[word] for word in words]
            #    #print "|".join(predict).encode("gb18030")
            target_input = autograd.Variable(target_input)
            if gpu_idx >= 0:
                target_input = target_input.cuda(gpu_idx)
            # B * 1
            target_input = target_input.view(-1, 1)

            # B * 1 * D
            trg_emb = self.embedding(target_input)
            # ( B * 1 * hD, B * 1 * hD)
            hidden = self.decoder.step(trg_emb, hidden, encoder_out, input_src_len)
            # B * 1 * V
            out = F.log_softmax(self.decoder2vocab(hidden[0]))
            word_lk = out.view(remaining_sents, beam_size, -1)
            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue
                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]
            if not active:
                break

            active_idx = torch.LongTensor([batch_idx[k] for k in active])
            if gpu_idx >= 0:
                active_idx = active_idx.cuda(gpu_idx)

            batch_idx = { beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                view = t.data.view(remaining_sents, beam_size, -1, self.trg_hidden_dim)
                step_size = view.size(2)
                if len(t.size()) == 3:
                    re = view.index_select(0, active_idx).view(len(active_idx)*beam_size,step_size, self.trg_hidden_dim)
                else:
                    re = view.index_select(0, active_idx).view(len(active_idx)*beam_size*step_size, self.trg_hidden_dim)
                if gpu_idx >= 0:
                    return autograd.Variable(re).cuda(gpu_idx)
                else:
                    return autograd.Variable(re)

            hidden = (update_active(hidden[0]), update_active(hidden[1]))
            encoder_out = update_active(encoder_out)
            remaining_sents = len(active)
        allHyp, allScores = [], []
        n_best = beam_size
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            allHyp += [hyps]
 
        return allHyp, allScores
        # rerank
        #res = [[], []]
        #for hyp, scores in zip(allHyp, allScores):
        #    # for each beatch
        #    batch_res = []
        #    for h, s in zip(hyp, scores):
        #        batch_res.append([h, s/len(h)])
        #    batch_res = sorted(batch_res, key = lambda x : x[1], reverse=True)
        #    hs = []
        #    ss = []
        #    for h, s in batch_res:
        #        hs.append(h)
        #        ss.append(s)
        #    res[0].append(hs)
        #    res[1].append(ss)

        #return res[0], res[1]



