import re
import codecs
import numpy as np


def create_dico(item_list, cur_map = {}):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    counter = len(cur_map)
    new_map = dict(cur_map)
    # idx from 1
    for items in item_list:
        for item in items:
            if item not in new_map:
                new_map[item] = counter
                counter += 1
    return new_map


def create_mapping(dico, vocabulary_size=2000):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), 
            key=lambda x: (-x[1], x[0]))[:vocabulary_size]
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def read_pre_training(emb_path):
    """
    Read pre-train word embeding
    The detail of this dataset can be found in the following link
    https://nlp.stanford.edu/projects/glove/ 
    """
    print('Preparing pre-train dictionary')
    emb_dictionary={}
    for line in codecs.open(emb_path, 'r', 'utf-8'):
        temp = line.split()
        emb_dictionary[temp[0]] = np.asarray(temp[1:], dtype= np.float16)
    return emb_dictionary


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)





