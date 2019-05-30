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

            # process premise and hypothesis
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

                    # if query_lemma[i].lower() != answer_lemma[j].lower():
                    if query_lemma[i] in self.concept_dict and answer_lemma[j] in self.concept_dict[query_lemma[i]]:
                        concept_qa[i, j, :] = self.concept_dict[query_lemma[i]][answer_lemma[j]]
                        # print("\rExample {}:\nThe premise is {}\nhypothesis is {}\nword one lemma is {}\nword two lemma is {}\nword one is {}\nword two is {}\nconcept is {}\n".format(
                        #         idx, premise, hypothesis, query_lemma[i], answer_lemma[j], widx, widy, concept_qa[i, j, :]))
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






class BertBatchDataLoader(Dataset):
    '''
     This data loader loads data.
     This supports any batch size.
    '''
    def __init__(self, fpath='', embd_dict=None, concept_dict=None, split='', max_len = 128, emd_dim = 300, num_concepts = 5):

        self.embd_dict = embd_dict
        self.concept_dict = concept_dict
        self.emd_dim = emd_dim
        self.num_concepts = num_concepts
        self.split = split

        # load in utterances
        everything = read_json_file(fpath)
        self.N = everything['split_size'][self.split]
        self.data = everything[split]
        self.max_len = max_len

        print("Done with loading data for %s split containing " % (self.split))
        print("Total: %d samples" % (self.N ))
        print('-' * 50)

    def __getitem__(self, idx):

        premise = self.data[idx]['premise']
        hypothesis = self.data[idx]['hypothesis']

        query = ['CLS'] + self.data[idx]['premise_tokens'] + ['EOS']
        query_segment = [0] * len(query)

        answer = self.data[idx]['hypothesis_tokens'] + ['EOS']
        answer_segment = [1] * len(answer)

        query_lemma = ['CLS'] + self.data[idx]['premise_lemmas'] + ['EOS']
        answer_lemma = self.data[idx]['hypothesis_lemmas'] + ['EOS']

        segment_ids = query_segment + answer_segment
        if len(segment_ids) < self.max_len:
            segment_ids = segment_ids + [0] * (self.max_len - len(segment_ids))
        else:
            segment_ids = segment_ids[:self.max_len]
        segment_ids = np.array(segment_ids)

        position_ids = np.concatenate([np.arange(len(query)), np.arange(len(answer))])
        if len(position_ids) < self.max_len:
            position_ids = np.concatenate([position_ids, np.zeros(self.max_len - len(position_ids))])
        else:
            position_ids = position_ids[:self.max_len]

        label = int(self.data[idx]['label'])
        concept_map = None
        query_length = len(query)

        # embedding vectors
        vecQA = np.zeros((self.max_len, self.emd_dim), dtype = np.float32)

        # concept vectors
        if self.concept_dict is not None:
            # map
            concept_map = np.zeros((self.max_len, self.max_len, self.num_concepts), dtype = np.float32) # [T, T, d]

            # process premise and hypothesis
            for i, widx in enumerate(query):
                if i >= self.max_len:
                    break
                try:
                    vecQA[i,:] = self.embd_dict[widx] # get the word embedding for premise
                except:
                    pass

                for j, widy in enumerate(answer):
                    if query_length + j >= self.max_len:
                        break

                    if query_lemma[i] in self.concept_dict and answer_lemma[j] in self.concept_dict[query_lemma[i]]:
                        concept_map[i, j+query_length, :] = self.concept_dict[query_lemma[i]][answer_lemma[j]]
                    if answer_lemma[j] in self.concept_dict and query_lemma[i] in self.concept_dict[answer_lemma[j]]:
                        concept_map[j+query_length, i, :] = self.concept_dict[answer_lemma[j]][query_lemma[i]]

        # create masks
        mask = build_mask(len(query)+len(answer), self.max_len)

        data  = {'qa':vecQA,
                 'segment_ids':segment_ids,
                 'position_ids':position_ids,
                 'concept':concept_map,
                 'mask':mask,
                 'label':label,
                 'qlength':len(query),
                 'alength':len(answer),
                 'length':len(query)+len(answer),
                 'qstr':premise,
                 'astr':hypothesis}

        return data

    def __len__(self):
        return self.N
