import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import wordnet as wn
import numpy as np

from misc.tokenization import *
import torch
import torch.optim as optim

import pickle
import argparse
from collections import Counter

from torchnlp.datasets import snli_dataset

import re

from tqdm import tqdm


def get_hyponyms(word):
    word = wn.synsets(word)
    if len(word) == 0:
        return []
    word = word[0]
    hyponyms = list(set([w for s in word.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))
    
    return hyponyms

def get_hypernyms(word):
    word = wn.synsets(word)
    if len(word) == 0:
        return []
    word = word[0]
    hypernyms = list(set([w for s in word.closure(lambda s:s.hypernyms()) for w in s.lemma_names()]))
    
    return hypernyms

def get_synonyms(word):
    word = wn.synsets(word)
    if len(word) == 0:
        return []
    synonyms = list(set([w for s in word for w in s.lemma_names()]))
    
    return synonyms

def get_vector(word1, word2):
    
    output = np.zeros(3)
    
    # check hyper/hypo-nym and synonyms
    synonyms = get_synonyms(word1)
    hypers = get_hypernyms(word1)
    hypos = get_hyponyms(word1)
    
    if word2 in synonyms:
        output[0] = 1
    if word2 in hypers:
        output[1] = 1
    if word2 in hypos:
        output[2] = 1
    
    return output


class Concept(object):
    """Cencept relation between words."""
    def __init__(self):
        self.word2rel = {}
        self.idx = 0

    def add_word(self, word, other=None, vec=None):
        if not other:
            self.word2rel[word] = np.zeros(3)
        else:
            if not word in self.word2rel:
                self.word2rel[word] = {other: vec}
                self.idx += 1
            else:
                self.word2rel[word][other] = vec

    def __call__(self, word, other):
        if not word in self.word2rel:
            return self.word2rel['<unk>']
        return self.word2rel[word][other]
    
    def __len__(self):
        return len(self.word2rel)
    
def prepare_vocab(dataset, threshold):
    
    counter = Counter()
    
    for t in tqdm(dataset):
        
        premise = t['premise']
        hypothesis = t['hypothesis']
        premise_tokens = nltk.word_tokenize(premise)
        hypothesis_tokens = nltk.word_tokenize(hypothesis)
        tokens = premise_tokens + hypothesis_tokens
        counter.update(tokens)
           
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
        
    # Create a vocab wrapper and add some special tokens.
    vocab = Concept()
    vocab.add_word('<unk>')
    
    # Add the words to the vocabulary.
    for word in tqdm(words):
        for other in words:
            vec = get_vector(word, other)
            vocab.add_word(word, other, vec)
            
    return vocab

if __name__=='__main__':
	train_data = snli_dataset(train=True)
	vocab = prepare_vocab(train_data, 3)