import torch
import torch.optim as optim
import numpy as np
from torchnlp.datasets import snli_dataset

import re
import nltk

def prepare_dataset(dataset):
    
    stats = {}
    
    for t in dataset:
        
        count_E = 0
        count_C = 0
        count_N = 0
        
        premise = t['premise']
        hypothesis = t['hypothesis']
        premise_tokens = nltk.word_tokenize(premise)
        hypothesis_tokens = nltk.word_tokenize(hypothesis)
        
        t['premise_tokens'] = premise_tokens
        t['hypothesis_tokens'] = hypothesis_tokens
        
        del t['premise_transitions'], t['hypothesis_transitions']
        

if __name__ == "__main__":
    
    train_data = snli_dataset(train=True)
    dev_data = snli_dataset(dev=True)
    test_data = snli_dataset(test=True)
    
    prepare_dataset(train_data)
    prepare_dataset(dev_data)
    prepare_dataset(test_data)
    
    print('Saving data ...')
    
    with open(os.path.join('../data', 'snli_train.json'), 'w+') as outfile:
        json.dump(train_data, outfile)
    with open(os.path.join('../data', 'snli_dev.json'), 'w+') as outfile:
        json.dump(dev_data, outfile)
    with open(os.path.join('../data', 'snli_test.json'), 'w+') as outfile:
        json.dump(test_data, outfile)