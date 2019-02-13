from __future__ import  print_function, division
import time
import torch
from torch.utils.data import Dataset
# from pytorch_pretrained_bert import BertTokenizer, BertModel
import numpy as np
import os
import json
import ntpath
import sys
from misc.utilities import read_json_file

def build_mask(len_data, max_len):


    if len_data > max_len:
        masked_data  = np.ones(max_len, dtype = np.float32)
    else:
        masked_data  = np.ones(len_data, dtype = np.float32)

    if len_data < max_len:
        masked_data = np.lib.pad(masked_data, (0, max_len - len_data), 'constant', constant_values=(0))

    return masked_data

class BatchDataLoader(Dataset):
    '''
     This data loader loads data.
     This supports any batch size.
    '''
    def __init__(self, fpath='', embd_dict=None, split='', max_q_len = 26, max_ans_len = 14, emd_dim = 300):

        self.embd_dict = embd_dict
        self.emd_dim = emd_dim
        self.split = split

        # load in utterances
        everything = read_json_file(fpath)
        self.N = everything['split_size'][self.split]
        self.data = everything[split]
        self.max_q_len = max_q_len
        self.max_ans_len = max_ans_len

        self.class_dict = {'entailment':0, 'neutral':1, 'contradiction':2}

        print("Done with loading data for %s split containing " % (self.split))
        print("Total: %d samples" % (self.N ))
        print("Entailment: %d" % (everything['n_entail'][split]))
        print("Neutral: %d" % (everything['n_neutral'][split]))
        print("Contradiction: %d" % (everything['n_contradiction'][split]))
        print('-' * 50)

    def __getitem__(self, idx):

        premise = self.data[idx]['premise']
        hypothesis = self.data[idx]['hypothesis']
        query = self.data[idx]['premise_tokens']
        answer = self.data[idx]['hypothesis_tokens']
        label = self.class_dict[self.data[idx]['label']]

        # embedding vectors
        vecQ1 = np.zeros((self.max_q_len, self.emd_dim), dtype = np.float32)
        vecQ2 = np.zeros((self.max_ans_len, self.emd_dim), dtype = np.float32)

        # process query (premise)
        for i, widx in enumerate(query):
            if i >= self.max_q_len:
                break
            vecQ1[i,:] = self.embd_dict[widx] # get the word embedding

        # process answer (hypothesis)
        for i, widx in enumerate(answer):
            if i >= self.max_ans_len:
                break
            vecQ2[i,:] = self.embd_dict[widx] # get the word embedding

        # create masks
        query_mask = build_mask(len(query), self.max_q_len)
        answer_mask = build_mask(len(answer), self.max_ans_len)

        data  = {'q1':vecQ1,
                 'q2':vecQ2,
                 'qmask':query_mask,
                 'amask':answer_mask,
                 'label':label,
                 'qlength':len(query),
                 'alength':len(answer),
                 'qstr':premise,
                 'astr':hypothesis}

        return data

    def __len__(self):
        return self.N
