import jsonlines, json, csv
import os, sys
import re
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
# nlp = spacy.load("en_core_web_sm")
from spacy.lang.en import English
tokenizer = English()

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

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class SnliProcessor(DataProcessor):
    """Processor for the SNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def get_labels(self):
        """See base class."""
        return CLASS_DICT

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(tqdm(lines, desc='snli')):
            if i == 0:
                continue
            premise = line[7]
            hypothesis = line[8]
            premise_tokens = [w.text for w in tokenizer(premise)]
            hypothesis_tokens = [w.text for w in tokenizer(hypothesis)]
            premise_lemmas = [w.lemma_ for w in tokenizer(premise)]
            hypothesis_lemmas = [w.lemma_ for w in tokenizer(hypothesis)]
            label = line[-1]

            if label in self.get_labels():
                label = self.get_labels()[label]
                examples.append(
                    {'premise':premise,
                     'hypothesis':hypothesis,
                     'premise_tokens':premise_tokens,
                     'hypothesis_tokens':hypothesis_tokens,
                     'premise_lemmas':premise_lemmas,
                     'hypothesis_lemmas':hypothesis_lemmas,
                     'label':label})
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")),
            "test_matched")

    def get_labels(self):
        """See base class."""
        return CLASS_DICT

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(tqdm(lines, desc='mnli')):
            if i == 0:
                continue
            premise = line[8]
            hypothesis = line[9]
            premise_tokens = [w.text for w in tokenizer(premise)]
            hypothesis_tokens = [w.text for w in tokenizer(hypothesis)]
            premise_lemmas = [w.lemma_ for w in tokenizer(premise)]
            hypothesis_lemmas = [w.lemma_ for w in tokenizer(hypothesis)]
            label = line[-1]

            if label in self.get_labels():
                label = self.get_labels()[label]
                examples.append(
                    {'premise':premise,
                     'hypothesis':hypothesis,
                     'premise_tokens':premise_tokens,
                     'hypothesis_tokens':hypothesis_tokens,
                     'premise_lemmas':premise_lemmas,
                     'hypothesis_lemmas':hypothesis_lemmas,
                     'label':label})
        return examples

if __name__ == "__main__":
    snli = SnliProcessor()
    mnli = MnliProcessor()
    snli_dir = './SNLI/'
    mnli_dir = './MNLI/'

    train_data = snli.get_train_examples(snli_dir) + mnli.get_train_examples(mnli_dir)
    dev_data = mnli.get_dev_examples(mnli_dir)
    test_data = mnli.get_test_examples(mnli_dir)
    adv_data = prepare_dataset('./snli/snli_1.0/data/dataset.jsonl', tokenizer)[6]
    data = {'train': train_data,
            'dev': dev_data,
            'test': test_data,
            'test_adv': adv_data,
            'split_size':{'train':len(train_data),
                          'dev':len(dev_data),
                          'test':len(test_data),
                          'test_adv':len(adv_data)}
            }

    print("Saving the data...\n")
    with open(os.path.join('./mnli', 'mnli_data.json'), 'w+') as outfile:
        json.dump(data, outfile)
