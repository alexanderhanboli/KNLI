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
import pdb

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
    def __init__(self, fpath='', embd_dict=None, concept_dict=None, split='', max_q_len = 90, 
                 max_ans_len = 70, emd_dim = 300, num_concepts = 5):

        self.embd_dict = embd_dict
        self.concept_dict = concept_dict
        self.emd_dim = emd_dim
        self.num_concepts = num_concepts
        self.split = split

        # load in utterances
        everything = read_json_file(fpath)
        self.N = everything['split_size'][self.split]
        self.data = everything[split]
        self.max_q_len = max_q_len
        self.max_ans_len = max_ans_len

        print("Done with loading data for %s split containing " % (self.split))
        print("Total: %d samples" % (self.N ))
        # print("Entailment: %d" % (everything['n_entail'][split]))
        # print("Neutral: %d" % (everything['n_neutral'][split]))
        # print("Contradiction: %d" % (everything['n_contradiction'][split]))
        print('-' * 50)

    def __getitem__(self, idx):

        premise = self.data[idx]['premise']
        hypothesis = self.data[idx]['hypothesis']
        query = self.data[idx]['premise_tokens']
        answer = self.data[idx]['hypothesis_tokens']
        query_lemma = self.data[idx]['premise_lemmas']
        answer_lemma = self.data[idx]['hypothesis_lemmas']
        label = int(self.data[idx]['label'])
        concept_qa = None
        concept_aq = None

        # embedding vectors
        vecQ1 = np.zeros((self.max_q_len, self.emd_dim), dtype = np.float32)
        vecQ2 = np.zeros((self.max_ans_len, self.emd_dim), dtype = np.float32)

        # concept vectors
        if self.concept_dict is not None:
            # q -> a
            concept_qa = np.zeros((self.max_q_len, self.max_ans_len, self.num_concepts), dtype = np.float32) # [T1, T2, d]
            # a -> q
            concept_aq = np.zeros((self.max_ans_len, self.max_q_len, self.num_concepts), dtype = np.float32) # [T2, T1, d]

        # process query (premise)
        for i, widx in enumerate(query):
            if i >= self.max_q_len:
                break

            try:
                vecQ1[i,:] = self.embd_dict[widx] # get the word embedding for premise
            except:
                pass

            for j, widy in enumerate(answer):
                if j >= self.max_ans_len:
                    break

                if i == 0:
                    try:
                        vecQ2[j,:] = self.embd_dict[widy] # get the word embedding for hypothesis
                    except:
                        pass
                        
                if self.concept_dict is not None:
                    if query_lemma[i] in self.concept_dict and answer_lemma[j] in self.concept_dict[query_lemma[i]]:
                        concept_qa[i, j, :] = self.concept_dict[query_lemma[i]][answer_lemma[j]]
                    if answer_lemma[j] in self.concept_dict and query_lemma[i] in self.concept_dict[answer_lemma[j]]:
                        concept_aq[j, i, :] = self.concept_dict[answer_lemma[j]][query_lemma[i]]

        # create masks
        query_mask = build_mask(len(query), self.max_q_len)
        answer_mask = build_mask(len(answer), self.max_ans_len)

        data  = {'q1':vecQ1,
                 'q2':vecQ2,
                 'concept_qa':concept_qa,
                 'concept_aq':concept_aq,
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
