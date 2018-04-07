#!/bin/bash
#encoding=utf8
'''
this file is created by burkun
date: 
'''


import codecs
import re
import HTMLParser
'''
基础字典
'''

html_parser = HTMLParser.HTMLParser()
default_t2s_name = "conf/t2s.dat"


def file_iter(file_path, encode="gb18030"):
    with codecs.open(file_path, "r", encoding=encode) as fin:
        for line in fin:
            line = line.strip("\n")
            yield line


def load_t2s_dic(file_path):
    dic = {}
    for line in file_iter(file_path):
        items = line.split(" ")
        if len(items) == 2:
            dic[items[0]] = items[1]
    return dic


def isalpha(word):
    if 'A' <= word <= 'z' or 'a' <= word <= 'z':
        return True
    return False

t2s_dict = load_t2s_dic(default_t2s_name)


'''
全角到半角转换
'''

class BCConvert(object):
    DBC_CHAR_START = 33 #半角感叹号
    DBC_CHAR_END = 126 #半角~
    #全角到半角的偏移量为65281，除了空格
    SBC_CHAR_START = 65218
    SBC_CHAR_END = 65374
    CONVERT_STEP = 65248 #相对的偏移量
    SBC_SPACE = 12288 #全角空格
    DBC_SPACE = ord(' ') #半角空格
    @staticmethod
    def bj2qj(text, ignor=set()):
        '''
        半角-全角转换
        '''
        if len(text.strip())== 0:
            return text
        res = []
        for idx in xrange(len(text)):
            code = ord(text[idx])
            if text[idx] not in ignor:
                if code == BCConvert.SBC_SPACE:
                    res.append(unichr(BCConvert.DBC_SPACE))
                elif  BCConvert.DBC_CHAR_START <= code <= BCConvert.DBC_CHAR_END:
                    res.append(unichr(code + BCConvert.CONVERT_STEP))
                else:
                    res.append(text[idx])
            else:
                res.append(text[idx])
        return "".join(res)
    @staticmethod
    def qj2bj(text, ignor = set()):
        '''
        全角-半角
        '''
        if len(text.strip()) == 0:
            return text
        res = []
        for idx in xrange(len(text)):
            code = ord(text[idx])
            if text[idx] not in ignor:
                if code == BCConvert.SBC_SPACE:
                    res.append(unichr(BCConvert.DBC_SPACE))
                elif BCConvert.SBC_CHAR_START <= code <= BCConvert.SBC_CHAR_END:
                    res.append(unichr(code - BCConvert.CONVERT_STEP))
                else:
                    res.append(text[idx])
            else:
                res.append(text[idx])
        return "".join(res)

class TextTools(object):
    '''
    清除括号
    '''
    @staticmethod
    def remove_bracket(text):
        text = re.sub("\(.+\)", "", text)
        text = re.sub(u"（.+）", "", text)
        return text
    '''
     繁简转换
    '''
    @staticmethod
    def fan2jian(text):
        res = ""
        for word in text:
            if word in t2s_dict:
               res += t2s_dict[word]
            else:
                res += word
        return res
    '''
    全角半角
    '''
    @staticmethod
    def quan2ban(text):
        return BCConvert.qj2bj(text)

    @staticmethod
    def norm_sen(text):
        #忽略大小写转换
        return TextTools.quan2ban(html_parser.unescape(TextTools.fan2jian(text)))

    @staticmethod
    def split_sent(line, min_len=50):
        if len(line) <= 50:
            return line
        re.split(u"!|,|;|\?|。|？|；|\t|\n\r", line)

    @staticmethod
    def filter_sent(line):
        if len(line) > 200 or len(line) < 3:
            return False
        return True


class WebTagFileProcesser(object):
    @staticmethod
    def feed_line(line):
        idx = 0
        while idx < len(line):
            if not (isalpha(line[idx]) or line[idx].isdigit()):
                yield line[idx]
                idx += 1
            res = ""
            while idx < len(line) and isalpha(line[idx]):
                res += line[idx]
                idx += 1
            if len(res) > 0:
                yield res
            res = ""
            while idx < len(line) and line[idx].isdigit():
                res += line[idx]
                idx += 1
            if len(res) > 0:
                yield res

    @staticmethod
    def convert_to_bie(f_path):
        res = []
        for line in file_iter(f_path):
            sent_tags = line.split("\t\t")
            if len(sent_tags) == 1:
                tags_items = []
            else:
                tags_items = sent_tags[1].split("\t")
            idx_map = {}
            for tag_item in tags_items:
                word_tag_begin_pos = tag_item.split("/")
                if len(word_tag_begin_pos) != 3:
                    continue
                word = word_tag_begin_pos[0]
                tag = word_tag_begin_pos[1]
                begin_pos = int(word_tag_begin_pos[2])
                for idx in range(len(word)):
                    if idx == 0:
                        idx_map[idx + begin_pos] = "B-" + tag.upper()
                    else:
                        idx_map[idx + begin_pos] = "I-" + tag.upper()
            offset = 0
            for word in WebTagFileProcesser.feed_line(sent_tags[0]):
                if offset in idx_map:
                    res.append([word, idx_map[offset]])
                else:
                    res.append([word, 'O'])
                offset += len(word)
            res.append("")
        return res

    @staticmethod
    def write_txt(f_path, fout_path):
        fout = codecs.open(fout_path, "w", encoding="utf-8")
        for line in file_iter(f_path):
            line.split("\t\t")
            fout.write(line.split("\t\t")[0] + "\n")
        fout.close()

    @staticmethod
    def write_bie(f_path, res):
        fout = codecs.open(f_path, "w", encoding="utf-8")
        for item in res:
            fout.write("\t".join(item) + "\n")
        fout.close()




class BiePostag():
    def __init__(self, min_len = 3, max_len = 100, split_words = u"!;?。？；\t\n\r"):
        self.split_words = set(iter(split_words))
        self.min_len = min_len
        self.max_len = max_len

    def feed_file(self, file_path, encode="gb18030"):
        fout = codecs.open(file_path, "r", encoding=encode, errors="ignore")
        res = []
        for line in fout:
            line = line.strip("\n")
            items = line.split("\t")
            if len(items) >= 2:
                items[0] = TextTools.norm_sen(items[0])
                for idx in range(1, len(items)):
                    items[idx] = items[idx].lower()
                res.append(items)
                if len(res) > self.max_len:
                    #TODO is break ne bound
                    yield res
                    res = []
                if items[0] in self.split_words:
                    if len(res) > self.min_len:
                        yield res
                    res = []
            else:
                if self.max_len >= len(res) >= self.min_len:
                    yield res
                res = []
        fout.close()

'''
分词，如果是单字，标记为S，如果是多个字，标记成BIE
'''

def seg2bie(line):
    line = line.strip("\n")
    if len(line) == 0:
        return None
    items = line.split("\t")
    res = []
    if len(item) == 1:
        res.append("S")
    elif len(item) == 2:
        res.append("B")
        res.append("E")
    else:
        res.append("B")
        for idx in range(1, len(item) - 1):
            res.append("I")
        res.append("E")
    return res



if __name__ == "__main__":
    pass
    pku = Pku2014Text()
    for line in pku.feed_file("../data/crf_data/test.bie.clean"):
        print line
