#! /usr/bin/env python
# -*- coding: gb18030 -*-
# vim:fenc=gb18030
#
# Copyright @ 2017 bookerbai <bookerbai@tencent.com>
#
# Distributed under terms of the Tencent license.

"""
1. language model
"""
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import codecs

class LMFileReader():
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
            line = line.strip()
            tokens = line.split("\t")
            for token in tokens:
                if token not in self.word_map:
                    self.word_map[token] = len(self.word_map)
        self.reset()
        return self.word_map

    def read(self, sample_num = -1):
        """
        -1标识全部读取
        """
        source = []
        sample_cnt = 0
        line = None
        while True:
            line = self.fin.readline()
            if not line:
                break
            sample_cnt += 1
            line = line.strip()
            tokens = line.split("\t")
            source_ids = [self.word_map["<s>"]]
            for item in tokens:
                if item in self.word_map:
                    source_ids.append(self.word_map[item])
                else:
                    source_ids.append(self.word_map["<unk>"])
            source_ids.append(self.word_map["</s>"])
            source.append(source_ids)
            if len(source) == sample_num and sample_num > 0:
                return source, not line
        return source, not line

class LMDataset():
    """save lm data
    """
    def __init__(self, file_path, word_map = None):
        self.data_buf = []
        self.reader = LMFileReader(file_path, word_map, "gb18030")

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
        return len(self.data_buf)

    def __getitem__(self, idx):
        return self.data_buf[idx]


class LMDataLoader():
    """seq2seq loader
    """
    def __init__(self, dataset, batch_size=128, shuff=True, pad=0):
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
        # get next, batch_res = [source, source]
        batch_res = []
        return_res = [[], []]
        batch_cnt = 0
        max_s_len = 0
        if len(self.dataset) > 0:
            for idx in self.idxs_iter:
                batch_cnt += 1
                st = self.dataset[idx]
                batch_res.append(st)
                if len(st) > max_s_len:
                    max_s_len = len(st)
                if batch_cnt == self.batch_size:
                    break
            batch_res = sorted(batch_res, key=lambda x: len(x), reverse=True)
            for item in batch_res:
                source = list(item)
                if len(source) < max_s_len:
                    source += ([self.pad] * (max_s_len - len(source)))
                return_res[1].append(len(item))
                return_res[0].append(source)
            self.counter += 1
            return return_res
        else:
            return None

if __name__ == "__main__":
    import sys
    dataset = LMDataset(sys.argv[1])
    dataloader = LMDataLoader(dataset, batch_size = 10)
    dataloader.start()
    counter = 0
    while True:
        nx = dataloader.get_next()
        if nx is None:
            break
        print nx
        counter += len(nx[0])
    dataloader.close()
    print counter
