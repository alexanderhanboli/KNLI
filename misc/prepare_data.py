import torch
import torch.optim as optim
import numpy as np
from torchnlp.datasets import snli_dataset

import os, json
import re
import nltk
nltk.download('punkt')

from tqdm import tqdm

import requests
import zipfile

def prepare_dataset(dataset):
    
    output = []
    stats = {}
    # counts of each class
    count_E = 0
    count_C = 0
    count_N = 0
    # lengths of sentences
    l_E = {'premise':[], 'hypothesis':[]}
    l_C = {'premise':[], 'hypothesis':[]}
    l_N = {'premise':[], 'hypothesis':[]}
    
    for t in tqdm(dataset, desc="preparing data"):
        
        premise = t['premise']
        hypothesis = t['hypothesis']
        premise_tokens = nltk.word_tokenize(premise)
        hypothesis_tokens = nltk.word_tokenize(hypothesis)
        
        t['premise_tokens'] = premise_tokens
        t['hypothesis_tokens'] = hypothesis_tokens
        
        if t['label'] == 'neutral':
            count_N += 1
            l_N['premise'].append(len(premise_tokens))
            l_N['hypothesis'].append(len(hypothesis_tokens))
        elif t['label'] == 'contradiction':
            count_C += 1
            l_C['premise'].append(len(premise_tokens))
            l_C['hypothesis'].append(len(hypothesis_tokens))
        elif t['label'] == 'entailment':
            count_E += 1
            l_E['premise'].append(len(premise_tokens))
            l_E['hypothesis'].append(len(hypothesis_tokens))
        else:
            continue
            
        del t['premise_transitions'], t['hypothesis_transitions']
        
        output.append(t)
        
    return count_E, count_C, count_N, l_E, l_C, l_N, output
        

if __name__ == "__main__":
    
    if not os.path.isfile('./data/snli/snli_data.json'):
        train_data = snli_dataset(train=True)
        dev_data = snli_dataset(dev=True)
        test_data = snli_dataset(test=True)

        tr_e, tr_c, tr_n, tr_le, tr_lc, tr_ln, train_data = prepare_dataset(train_data)
        dev_e, dev_c, dev_n, dev_le, dev_lc, dev_ln, dev_data = prepare_dataset(dev_data)
        test_e, test_c, test_n, test_le, test_lc, test_ln, test_data = prepare_dataset(test_data)

        data = {'train': list(train_data), 'dev': list(dev_data), 'test': list(test_data), 
            'n_entail': {'train': tr_e, 'dev':dev_e, 'test':test_e},
            'n_contradiction': {'train':tr_c, 'dev':dev_c, 'test':test_c}, 
            'n_neutral': {'train':tr_n, 'dev':dev_n, 'test':test_n}, 
            'len_entail': {'train': tr_le, 'dev':dev_le, 'test':test_le},
            'len_contradiction': {'train':tr_lc, 'dev':dev_lc, 'test':test_lc}, 
            'len_neutral': {'train':tr_ln, 'dev':dev_ln, 'test':test_ln}, 
            'split_size': {'train':tr_e + tr_c + tr_n, 'dev':dev_e+dev_c+dev_n, 'test':test_e+test_c+test_n}}

        print('Saving data ...')

        with open(os.path.join('data', 'snli_data.json'), 'w') as outfile:
            json.dump(data, outfile)
        
    print('Loading in word embeddings ...')
    if not os.path.isfile('./data/wiki.en.bin'):
        r = requests.get('https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip')
        assert r.status_code == requests.codes.ok
        zip_ref = zipfile.ZipFile('./data/wiki.en.zip', 'r')
        zip_ref.extractall('./data/')
        zip_ref.close()