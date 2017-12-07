#! /usr/bin/env python
# -*- coding: gb18030 -*-
# vim:fenc=gb18030
#

"""
extend CRF's feature
1. include extrace crf feature to feature idx
2. generate feature to feature idx

"""
import re
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

class SentTable():
    def __init__(self, col_size = 1):
        self.data = []
        self.col_size = col_size

    def add_row(self, row):
        if len(row) == self.col_size:
            self.data.append(row)
        else:
            raise Exception("can not insert row, col size not match")

    def build(self, sent_list):
        for row in sent_list:
            self.add_row(row[0:-1])

    def get_ele(self, row, col):
        return self.data[row][col]

    def __len__(self):
        return len(self.data)

    def get_col_size(self):
        return self.col_size

class FeatureTemplate():
    START_PADDING=["_B-3", "_B-2", "_B-1"]
    END_PADDING=["_B+1", "_B+2", "_B+3"]
    def __init__(self, feature_type_name):
        self.feature_type_name = feature_type_name
        self.table_idxs = []
        self.marker = set()

    def add_feature(self, row, col):
        if row > 3 or row < -3:
            raise Exception("feature can only support in [-3, 3] range")
        set_key = self.feature_type_name + str(row) +"-"+ str(col)
        if set_key not in self.marker:
            self.table_idxs.append([row,col])

    def __len__(self):
        return len(self.table_idxs)

    def apply_rule(self, sent_table, pos):
        feature_str = []
        sent_len = len(sent_table)
        col_len = sent_table.get_col_size()
        for t_idx in self.table_idxs:
            row = pos + t_idx[0]
            col = t_idx[1]
            if row < 0:
                feature_part = self.START_PADDING[row]
            elif row >= sent_len:
                feature_part = self.END_PADDING[row-sent_len]
            else:
                feature_part = sent_table.get_ele(row, col)
            feature_str.append(feature_part)
        return self.feature_type_name + ":" + "/".join(feature_str)

class FeatureHelper():
    template_re = re.compile("%x\[([\-\+\d]+),([\-\+\d]+)\]")
    def __init__(self, feature_path):
        self.total_template = self._load_feature(feature_path)

    def _load_feature(self, feature_path):
        total_feat = []
        fin = open(feature_path)
        for line in fin:
            line = line.strip()
            if line.startswith("#"):
                continue
            else:
                items = line.split(":")
                if len(items) == 2:
                    matches = self.template_re.finditer(line)
                    ft = FeatureTemplate(items[0])
                    for m in matches:
                        ft.add_feature(int(m.group(1)), int(m.group(2)))
                    if len(ft) > 0:
                        total_feat.append(ft)
        fin.close()
        return total_feat

    def apply_rule(self, sent_table, pos):
        features = []
        for template in self.total_template:
            features.append(template.apply_rule(sent_table, pos))
        return features

    def get_template_num(self):
        return len(self.total_template)

class PureDataSet(Dataset):
    def __init__(self, text_list, tag_list, feature_matrix, feature_size = None, tag_size = None):
        super(PureDataSet, self).__init__()
        #N * S
        self.text_list = text_list
        #N * S
        self.tag_list = tag_list
        #N * S * TAG * FeatureTempSize
        self.feature_matrix = feature_matrix
        self.feature_size = feature_size
        self.tag_size = tag_size

    def __getitem__(self, index):
        return self.text_list[index], self.tag_list[index], self.feature_matrix[index]

    def __len__(self):
        return len(self.text_list)

    def get_tag_size(self):
        return self.tag_size

    def get_feature_size(self):
        return self.feature_size


class PureCrfDataLoader():
    def __init__(self, dataset, pad_val=0, batch_size=1, shuffle=True, gpu_idx = None):
        self.shuffle = shuffle
        self.data_len = len(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.pad_val = pad_val
        self.gpu_idx = gpu_idx
        if shuffle:
            self.idxs_iter = iter(torch.randperm(self.data_len))
        else:
            self.idxs_iter = iter(xrange(0, self.data_len))

    def reset(self):
        if self.shuffle:
            self.idxs_iter = iter(torch.randperm(self.data_len))
        else:
            self.idxs_iter = iter(xrange(0, self.data_len))

    def get_next(self, transpos = True):
        list_data = []
        counter = 0
        for true_idx in self.idxs_iter:
            list_data.append(self.dataset[true_idx])
            counter += 1
            if counter >= self.batch_size:
                break
        if len(list_data) == 0:
            return None
        list_data = sorted(list_data, key=lambda x: len(x[0]), reverse=True)
        max_len = len(list_data[0][0])
        lengths = []
        input_words = []
        tags = []
        input_features = []

        for item in list_data:
            # deep copy
            input_line = list(item[0])
            input_tag = list(item[1])
            input_feature = list(item[2])
            item_word_len = len(item[0])
            lengths.append(item_word_len)
            if item_word_len < max_len:
                input_line.extend([self.pad_val] * (max_len - item_word_len))
                input_feature.extend([[self.pad_val] * self.dataset.get_feature_size()] * (max_len - item_word_len))
                input_tag.extend([self.pad_val] * (max_len - item_word_len))
            input_words.append(input_line)
            tags.append(input_tag)
            input_features.append(input_feature)

        if self.gpu_idx is not None and self.gpu_idx >= 0:
            with torch.cuda.device(self.gpu_idx):
                tags = torch.cuda.LongTensor(tags)
                inputs = torch.cuda.LongTensor(input_words)
                features = torch.cuda.LongTensor(input_features)
        else:
            tags = torch.LongTensor(tags)
            inputs = torch.LongTensor(input_words)
            features = torch.LongTensor(input_features)

        lengths = torch.LongTensor(lengths)
        #inputs = inputs.view(counter, -1)
        #tags = tags.view(counter, -1)
        if transpos:
            inputs = inputs.t()
            tags = tags.t()
            features = features.transpose(0, 1)
            features = features.contiguous()
            tags = tags.contiguous()
            inputs = inputs.contiguous()
        return inputs, tags, features, lengths

class TagDictMgr():
    def __init__(self, list_tags):
        self.tag_dict = {}
        self.rev_tag_dict = {}
        counter = 0
        for ele in list_tags:
            self.tag_dict[ele.lower()] = counter
            self.rev_tag_dict[counter] = ele
            counter += 1

    @staticmethod
    def build_from(tag_dict):
        items = sorted(tag_dict.items(), key=lambda x : x[1])
        tags = [item[0] for item in items]
        return TagDictMgr(tags)

    def get_tag_pos(self, tag):
        tag = tag.lower()
        return self.tag_dict[tag]

    def get_tag_dict(self):
        return self.tag_dict

    def get_pos_tag(self, pos):
        return self.rev_tag_dict[pos]

    def get_tag_size(self):
        return len(self.tag_dict)

class CrfDataSet(Dataset):
    def __init__(self, text_list, tag_list, pos_list = None):
        super(CrfDataSet, self).__init__()
        self.text_list = text_list
        self.tag_list = tag_list
        self.pos_list = pos_list

    def __getitem__(self, index):
        if self.pos_list is None:
            return self.text_list[index], self.tag_list[index]
        else:
            return self.text_list[index], self.tag_list[index], self.pos_list[index]

    def __len__(self):
        return len(self.text_list)


class CrfDataLoader():
    def __init__(self, dataset, pad_val = 0, batch_size=1, shuffle=True, gpu_idx = None):
        self.shuffle = shuffle
        self.data_len = len(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.pad_val = pad_val
        self.gpu_idx = gpu_idx
        if shuffle:
            self.idxs_iter = iter(torch.randperm(self.data_len))
        else:
            self.idxs_iter = iter(xrange(0, self.data_len))

    def reset(self):
        if self.shuffle:
            self.idxs_iter = iter(torch.randperm(self.data_len))
        else:
            self.idxs_iter = iter(xrange(0, self.data_len))

    def get_next(self, transpos = True):
        list_data = []
        counter = 0
        for true_idx in self.idxs_iter:
            list_data.append(self.dataset[true_idx])
            counter += 1
            if counter >= self.batch_size:
                break
        if len(list_data) == 0:
            return None
        list_data = sorted(list_data, key=lambda x: len(x[0]), reverse=True)
        inputs = []
        tags = []
        pos = []
        position = []
        lengths = []
        max_len = len(list_data[0][0])
        for item in list_data:
            # deep copy
            input_line = list(item[0])
            input_tags = list(item[1])
            input_position = [idx for idx in range(len(item[0]))]
            if len(item) > 2:
                input_pos = list(item[2])
            else:
                input_pos = None
            item_word_len = len(item[0])
            lengths.append(item_word_len)
            if item_word_len < max_len:
                input_line.extend([self.pad_val] * (max_len - item_word_len))
                input_tags.extend([self.pad_val] * (max_len - item_word_len))
                input_position.extend([self.pad_val] * (max_len - item_word_len))
                if input_pos is not None:
                    input_pos.extend([self.pad_val] * (max_len - item_word_len))
            inputs.append(input_line)
            tags.append(input_tags)
            position.append(input_position)
            if input_pos is not None:
                pos.append(input_pos)
        if self.gpu_idx is not None and self.gpu_idx >= 0:
            with torch.cuda.device(self.gpu_idx):
                tags = torch.cuda.LongTensor(tags)
                inputs = torch.cuda.LongTensor(inputs)
                position = torch.cuda.LongTensor(position)
                if len(pos) > 0:
                    pos = torch.cuda.LongTensor(pos)
                # lengths = torch.cuda.LongTensor(lengths)
        else:
            tags = torch.LongTensor(tags)
            inputs = torch.LongTensor(inputs)
            position = torch.LongTensor(position)
            if len(pos) > 0:
                pos = torch.LongTensor(pos)
        lengths = torch.LongTensor(lengths)
        position = position.view(counter, -1)
        inputs = inputs.view(counter, -1)
        tags = tags.view(counter, -1)
        if transpos:
            inputs = inputs.t()
            tags = tags.t()
            position = position.t()
        if len(pos) > 0:
            pos = pos.view(counter, -1)
            if transpos:
                pos = pos.t()
            return inputs, tags, pos, position, lengths
        else:
            return inputs, tags, None, position, lengths


