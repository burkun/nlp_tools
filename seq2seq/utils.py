#! /usr/bin/env python
# -*- coding: gb18030 -*-
# vim:fenc=gb18030

import json, codecs
import torch
import numpy as np
import math
from collections import Counter

def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in xrange(1, 3):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in xrange(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in xrange(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""

    print stats
    print filter(lambda x : x == 0, stats)
    if len(filter(lambda x: x == 0, stats)) > 0:
        return 0
    print "@@", stats
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


class ModelConfig():
    """¼ÓÔØmodelµÄ²ÎÊý
    """
    def __init__(self, config_path):
        self.cfg = json.load(open(config_path, "r"))

    def get_cfg(self):
        return self.cfg

    def __str__(self):
        return json.dumps(self.cfg, indent=4)

class FileReader():
    def __init__(self, file_path, word_map = None, encode="gb18030"):
        self.fin = None
        self.file_path = file_path 
        self.encode = encode
        self.inited_word = True
        if word_map is None:
            self.word_map = {"<s>" : 1, "</s>": 2, "<unk>": 3, "<pad>":0}
            self.inited_word = False
        else:
            self.word_map = word_map
            self.inited_word = True

    def open(self):
        self.fin = codecs.open(self.file_path, "r", encoding=self.encode)

    def close(self):
        if self.fin is not None:
            self.fin.close()

    def reset(self):
        if self.fin is None or self.fin.closed:
            return
        self.fin.seek(0)

    def init_wordmap(self):
        if self.inited_word:
            return self.word_map
        for line in self.fin:
            line = line.strip("\n")
            items = line.split("\t\t")
            for item in items:
                tokens = item.split("\t")
                for token in tokens:
                    if token not in self.word_map:
                        self.word_map[token] = len(self.word_map)
        self.reset()
        return self.word_map

    def read(self, sample_num = -1):
        """ -1 ±íÊ¾¶ÁÈ¡È«²¿
        """
        #res[0] = src, #res[1] = target
        source = []
        target = []
        sample_cnt = 0
        line = None
        while True:
            line = self.fin.readline()
            if not line:
                break
            sample_cnt += 1
            line = line.strip("\n")
            items = line.split("\t\t")
            # items[0] = source
            # items[1] = target, items[2] = target
            if len(items) > 2:
                #source_ids = [self.word_map["<s>"]]
                source_ids = []
                for item in items[0].split("\t"):
                    if item in self.word_map:
                        source_ids.append(self.word_map[item])
                    else:
                        source_ids.append(self.word_map["<unk>"])
                #source_ids.append(self.word_map["</s>"])
                for i in range(2, len(items)):
                    target_ids = [self.word_map["<s>"]]
                    for item in items[i].split("\t"):
                        if item in self.word_map:
                            target_ids.append(self.word_map[item])
                        else:
                            target_ids.append(self.word_map["<unk>"])
                    target_ids.append(self.word_map["</s>"])
                    source.append(source_ids)
                    target.append(target_ids)
                if len(source) == sample_num and sample_num > 0:
                    return [source, target], not line
        return [source, target], not line

class Seq2SeqDataset():
    """Éú³Édataset
    """
    def __init__(self, file_path, word_map = None):
        self.data_buf = []
        self.reader = FileReader(file_path, word_map, "gb18030")

    def open(self):
        self.reader.open()

    def close(self):
        self.reader.close()

    def init_wordmap(self):
        return self.reader.init_wordmap()

    def read_next(self, sample_num = -1):
        self.data_buf = []
        self.data_buf, is_end = self.reader.read(sample_num)
        return not is_end

    def reset(self):
        self.reader.reset()

    def __len__(self):
        return len(self.data_buf[0])

    def __getitem__(self, idx):
        return self.data_buf[0][idx], self.data_buf[1][idx]


class Seq2SeqDataLoader():
    """seq2seq batch Êý¾ÝÉú³É
    """
    def __init__(self, dataset, batch_size=1, shuff=True, pad=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.buffed_size = 20000
        self.sample_num = batch_size * self.buffed_size
        self.counter = 0
        self.pad = pad
        self.shuff = shuff
        self.word_map = None

    def reset(self):
        self.dataset.reset()
        self.counter = 0
        self.dataset.read_next(self.sample_num)
        self.__shuffle()

    def close(self):
        self.dataset.close()

    def start(self):
        self.counter = 0
        self.dataset.open()
        self.word_map = self.dataset.init_wordmap()

        self.dataset.read_next(self.sample_num)
        self.__shuffle()

    def __shuffle(self):
        if self.shuff:
            self.idxs_iter = iter(torch.randperm(len(self.dataset)))
        else:
            self.idxs_iter = iter(xrange(0, len(self.dataset)))

    def get_next(self):
        if self.counter*self.batch_size >= len(self.dataset):
            self.counter = 0
            self.dataset.read_next(self.sample_num)
            if len(self.dataset) == 0:
                return None
            self.__shuffle()
        # get next
        batch_res = []
        return_res = [[], [], []]
        batch_cnt = 0
        max_s_len = 0
        max_t_len = 0
        if len(self.dataset) > 0:
            for idx in self.idxs_iter:
                batch_cnt += 1
                st = self.dataset[idx]
                batch_res.append(st)
                if len(st[0]) > max_s_len:
                    max_s_len = len(st[0])
                if len(st[1]) > max_t_len:
                    max_t_len = len(st[1])
                if batch_cnt == self.batch_size:
                    break
            batch_res = sorted(batch_res, key=lambda x: len(x[0]), reverse=True)
            for item in batch_res:
                source = list(item[0])
                target = list(item[1])
                #print source
                #print target
                if len(source) < max_s_len:
                    source += ([self.pad] * (max_s_len - len(source)))
                if len(target) < max_t_len:
                    target += ([self.pad] * (max_t_len - len(target)))
                return_res[2].append(len(item[0]))
                return_res[0].append(source)
                return_res[1].append(target)
            self.counter += 1
            return return_res
        else:
            return None


if __name__ == "__main__":
    import sys

    dataset = Seq2SeqDataset(sys.argv[1])

    loader = Seq2SeqDataLoader(dataset, batch_size = 10, shuff = True, pad = 0)
    loader.start()
    word_map = loader.word_map
    counter = 0
    while True:
        nx = loader.get_next()
        if nx is not None:
            counter += len(nx[0])
        if nx is None:
            loader.reset()
            print counter
            break
            counter = 0
    loader.close()

