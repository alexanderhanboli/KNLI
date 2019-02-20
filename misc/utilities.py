from __future__ import  print_function, division
import time
import sys
import math
import ntpath
import os
import json
import ntpath
import numpy as np
import fastText
import torch.nn as nn

class Preload_embedding():

    def __init__(self, emd_file='data/wiki.en.bin', emb_context_file = '', emb_type ='generic'):

        print("Loading embedding file {}".format(emd_file))
        start = time.clock()
        self.model = fastText.load_model(emd_file)
        if emb_context_file:
            self.model_context = fastText.load_model(emb_context_file)
        else:
            self.model_context = None
        duration = time.clock() - start
        self.emb_type = emb_type
        print('Finished reading embeddings ', duration, 's')
        print('-' * 50)

    def __getitem__(self, token):
        if self.emb_type == 'context':
            return self.model_context.get_word_vector(token)
        elif self.emb_type == 'both':
            return np.concatenate((self.model.get_word_vector(token),self.model_context.get_word_vector(token)))
        else:
            return self.model.get_word_vector(token)


    def __contains__(self, token):
        if self.model.get_word_id(token) == -1:
            return False
        else:
            return True

    def get_without_subwords(self, token):
        if self.model.get_word_id(token) != -1:
            return self.model.get_word_vector(token)
        else:
            raise KeyError(token)

    def get_dimension(self):
        return self.model.get_dimension()

def create_dir(directory):

    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            raise ValueError(e)

def dump_to_json(path, data):
    '''
      Write json file
    '''
    with open(path, 'w') as f:
        json.dump(data, f)

def read_json_file(input_json):
    ## load the json file
    file_info = json.load(open(input_json, 'r'))

    return file_info

def get_fname_from_path(f):
    '''
     input:
           '/Users/user/logs/check_points/mmmxm_dummy_B32_H5_D1_best.pt'
     output:
           'mmmxm_dummy_B32_H5_D1_best.pt'
    '''
    return ntpath.basename(f)

def timeSince(since):
    now =  time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data

def initialize_weights(model):
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

def orthogonal_weights(model):
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        if p.dim() > 1:
            nn.init.orthogonal_(p)
