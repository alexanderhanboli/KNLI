import jsonlines, json
import os
import re
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
# nlp = spacy.load("en_core_web_sm")
from spacy.lang.en import English
nlp = English()

from tqdm import tqdm

CLASS_DICT = {'entailment':0, 'neutral':1, 'contradiction':2}

def prepare_dataset(dataset, tokenizer):
    
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
    
    with jsonlines.open(dataset) as reader:
        for t in tqdm(reader, desc="loading "+dataset):
            tmp = {}
            premise = t['sentence1']
            hypothesis = t['sentence2']
            premise_tokens = [w.text for w in tokenizer(premise)]
            hypothesis_tokens = [w.text for w in tokenizer(hypothesis)]
            premise_lemmas = [w.lemma_ for w in tokenizer(premise)]
            hypothesis_lemmas = [w.lemma_ for w in tokenizer(hypothesis)]

            tmp['premise'] = premise
            tmp['hypothesis'] = hypothesis
            tmp['premise_tokens'] = premise_tokens
            tmp['hypothesis_tokens'] = hypothesis_tokens
            tmp['premise_lemmas'] = premise_lemmas
            tmp['hypothesis_lemmas'] = hypothesis_lemmas
            
            if t['gold_label'] == 'neutral':
                count_N += 1
                l_N['premise'].append(len(premise_tokens))
                l_N['hypothesis'].append(len(hypothesis_tokens))
            elif t['gold_label'] == 'contradiction':
                count_C += 1
                l_C['premise'].append(len(premise_tokens))
                l_C['hypothesis'].append(len(hypothesis_tokens))
            elif t['gold_label'] == 'entailment':
                count_E += 1
                l_E['premise'].append(len(premise_tokens))
                l_E['hypothesis'].append(len(hypothesis_tokens))
            else:
                continue

            tmp['label'] = CLASS_DICT[t['gold_label']]

            output.append(tmp)
        
    return count_E, count_C, count_N, l_E, l_C, l_N, output

tr_e, tr_c, tr_n, tr_le, tr_lc, tr_ln, train_data = prepare_dataset('./snli/snli_1.0/snli_1.0_train.jsonl', nlp)
dev_e, dev_c, dev_n, dev_le, dev_lc, dev_ln, dev_data = prepare_dataset('./snli/snli_1.0/snli_1.0_dev.jsonl', nlp)
test_e, test_c, test_n, test_le, test_lc, test_ln, test_data = prepare_dataset('./snli/snli_1.0/snli_1.0_test.jsonl', nlp)
testadv_e, testadv_c, testadv_n, testadv_le, testadv_lc, testadv_ln, testadv_data = prepare_dataset('./snli/snli_1.0/data/dataset.jsonl', nlp)

data = {'train': list(train_data), 'dev': list(dev_data), 'test': list(test_data), 'test_adv': list(testadv_data),
        'n_entail': {'train': tr_e, 'dev':dev_e, 'test':test_e, 'test_adv':testadv_e},
        'n_contradiction': {'train':tr_c, 'dev':dev_c, 'test':test_c, 'test_adv':testadv_c}, 
        'n_neutral': {'train':tr_n, 'dev':dev_n, 'test':test_n, 'test_adv':testadv_n}, 
        'len_entail': {'train': tr_le, 'dev':dev_le, 'test':test_le, 'test_adv':testadv_le},
        'len_contradiction': {'train':tr_lc, 'dev':dev_lc, 'test':test_lc, 'test_adv':testadv_lc}, 
        'len_neutral': {'train':tr_ln, 'dev':dev_ln, 'test':test_ln, 'test_adv':testadv_ln}, 
        'split_size': {'train':tr_e + tr_c + tr_n, 'dev':dev_e+dev_c+dev_n, 'test':test_e+test_c+test_n, 'test_adv':testadv_e+testadv_c+testadv_n}}

print("Saving the data...\n")
with open(os.path.join('./snli', 'snli_data.json'), 'w+') as outfile:
    json.dump(data, outfile)

# print out some statistics
# training
print("\nTraining data stats...\n")
print("average length of premise: entail: {}, contradiction: {}, neutral: {}".format(np.mean(data['len_entail']['train']['premise']), np.mean(data['len_contradiction']['train']['premise']), np.mean(data['len_neutral']['train']['premise'])))
print("max length of premise: entail: {}, contradiction: {}, neutral: {}".format(np.max(data['len_entail']['train']['premise']), np.max(data['len_contradiction']['train']['premise']), np.max(data['len_neutral']['train']['premise'])))
print("average length of hypothesis: entail: {}, contradiction: {}, neutral: {}".format(np.mean(data['len_entail']['train']['hypothesis']), np.mean(data['len_contradiction']['train']['hypothesis']), np.mean(data['len_neutral']['train']['hypothesis'])))
print("max length of hypothesis: entail: {}, contradiction: {}, neutral: {}".format(np.max(data['len_entail']['train']['hypothesis']), np.max(data['len_contradiction']['train']['hypothesis']), np.max(data['len_neutral']['train']['hypothesis'])))

# validation
print("\nValidation data stats...\n")
print("average length of premise: entail: {}, contradiction: {}, neutral: {}".format(np.mean(data['len_entail']['dev']['premise']), np.mean(data['len_contradiction']['dev']['premise']), np.mean(data['len_neutral']['dev']['premise'])))
print("max length of premise: entail: {}, contradiction: {}, neutral: {}".format(np.max(data['len_entail']['dev']['premise']), np.max(data['len_contradiction']['dev']['premise']), np.max(data['len_neutral']['dev']['premise'])))
print("average length of hypothesis: entail: {}, contradiction: {}, neutral: {}".format(np.mean(data['len_entail']['dev']['hypothesis']), np.mean(data['len_contradiction']['dev']['hypothesis']), np.mean(data['len_neutral']['dev']['hypothesis'])))
print("max length of hypothesis: entail: {}, contradiction: {}, neutral: {}".format(np.max(data['len_entail']['dev']['hypothesis']), np.max(data['len_contradiction']['dev']['hypothesis']), np.max(data['len_neutral']['dev']['hypothesis'])))

# training
print("\nTest data stats...\n")
print("average length of premise: entail: {}, contradiction: {}, neutral: {}".format(np.mean(data['len_entail']['test']['premise']), np.mean(data['len_contradiction']['test']['premise']), np.mean(data['len_neutral']['test']['premise'])))
print("max length of premise: entail: {}, contradiction: {}, neutral: {}".format(np.max(data['len_entail']['test']['premise']), np.max(data['len_contradiction']['test']['premise']), np.max(data['len_neutral']['test']['premise'])))
print("average length of hypothesis: entail: {}, contradiction: {}, neutral: {}".format(np.mean(data['len_entail']['test']['hypothesis']), np.mean(data['len_contradiction']['test']['hypothesis']), np.mean(data['len_neutral']['test']['hypothesis'])))
print("max length of hypothesis: entail: {}, contradiction: {}, neutral: {}".format(np.max(data['len_entail']['test']['hypothesis']), np.max(data['len_contradiction']['test']['hypothesis']), np.max(data['len_neutral']['test']['hypothesis'])))