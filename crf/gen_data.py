#! /usr/bin/env python
# -*- coding: gb18030 -*-
# vim:fenc=gb18030
#
# Copyright @ 2017

"""
¼ÓÔØÑµÁ·Êý¾Ý£¬Éú³Éfeature
"""
import codecs
import sys
import os
from crf_utils import *

def gen_sent_feature(feature_path, feature_dict = {}):
    fh = FeatureHelper("template")
    fin = codecs.open(feature_path, "r", encoding="gb18030")
    sent = SentTable()
    tag_set = set()
    word_dict = dict()
    word_dict["UNK"] = 0
    word_counter = 1
    log_counter = 0
    feature_id = 1
    for line in fin:
        line = line.strip("\n")
        items = line.split("\t")
        if len(items) == 2:
            sent.add_row(items[0:-1])
            tag_set.add(items[-1].lower())
            if items[0] not in word_dict:
                word_dict[items[0]] = word_counter
                word_counter += 1
        elif len(line) == 0:
            os.write(1, "cur sent num = %d\r" % log_counter)
            sys.stdout.flush()
            log_counter += 1
            for idx in range(len(sent)):
                features = fh.apply_rule(sent, idx)
                for f in features:
                    if f in feature_dict:
                        feature_dict[f][1] += 1
                    else:
                        feature_dict[f] = [feature_id, 1]
                        feature_id += 1
            sent = SentTable()
    if len(sent) > 0:
       for idx in range(len(sent)):
           features = fh.apply_rule(sent, idx)
           for f in features:
               if f in feature_dict:
                   feature_dict[f][1] += 1
               else:
                   feature_dict[f] = [feature_id, 1]
    fin.close()
    return feature_dict, tag_set, word_dict

def save_features(file_path, feature_dict, tag_set):
    tag_size = len(tag_set)
    # feature ´Ó1¿ªÊ¼£¬0Áô¸øunk
    fout = codecs.open(file_path, "w", encoding="gb18030")
    fout.write("##feature size = %d, tag size = %d##\n" % (len(feature_dict) * len(tag_set), len(tag_set)))
    for feature in feature_dict:
        fout.write(feature + "\t" + str(feature_dict[feature][0]) + "\t" + str(feature_dict[feature][1]) + "\n")
    fout.close()

def load_features(file_path):
    fin = codecs.open(file_path, "r", encoding="gb18030")
    feature_dict = {}
    for line in fin:
        line = line.strip()
        if not line.startswith("#"):
            items = line.split("\t")
            if len(items) == 3:
                feature_dict[items[0]] = [int(items[1]), int(items[2])]
    return feature_dict


def create_train_data(feature_dict, tag_set, word_dict, train_path, target_path):
    fh = FeatureHelper("template")
    fin = codecs.open(train_path, "r", encoding="gb18030")
    fout = codecs.open(target_path, "w", encoding="gb18030")
    fout.write("%d;%d;%s\n" % (len(feature_dict) * len(tag_set), len(word_dict), ",".join(tag_set)))
    total_sent = []
    sent = SentTable()
    tags = []
    for key in word_dict:
        fout.write(key + "\t" + str(word_dict[key]) + "\n")
    fout.write("\n")
    print "read train file..."
    for line in fin:
        line = line.strip("\n")
        items = line.split("\t")
        if len(items) == 2:
            sent.add_row(items[0:-1])
            tags.append(items[-1])
        elif len(line) == 0:
            if len(sent) > 0:
                total_sent.append([sent, tags])
            sent =SentTable()
            tags = []
    fin.close()
    if len(sent) > 0:
        total_sent.append([sent, tags])
    # sort input line
    # print "total sent = %d, sort train file...(for speed up)" % len(total_sent)
    # total_sent = sorted(total_sent, key=lambda x : len(x[0]), reverse=True)
    # print "write train feature file..."
    for sent, tags in total_sent:
        for idx in range(len(sent)):
            features = fh.apply_rule(sent, idx)
            feature_idxs = []
            for f in features:
                if f in feature_dict:
                    feature_idxs.append(str(feature_dict[f][0]))
                else:
                    feature_idxs.append("0")
            if len(feature_idxs) > 0:
                fout.write(sent.get_ele(idx,0) + "\t" + ",".join(feature_idxs) + "\t" + tags[idx].lower() + "\n")
        fout.write("\n")
    fout.close()


if __name__ == "__main__":
    # arg1 = train.raw
    # arg2 = feature path
    # arg3 = train.new
    # arg4 = old_feature
    if len(sys.argv) > 4:
        print "load old features"
        old_features = load_features(sys.argv[4])
    else:
        old_features = {}
    fd, ts, word_dict = gen_sent_feature(sys.argv[1], old_features)
    print "save features.."
    sys.stdout.flush()
    save_features(sys.argv[2], fd, ts)
    fd = load_features(sys.argv[2])
    print "gen train data.."
    create_train_data(fd, ts, word_dict, sys.argv[1], sys.argv[3])
