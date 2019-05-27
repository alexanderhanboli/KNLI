from __future__ import  print_function, division
import torch.utils.data as data_utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import os
import json
import pickle as pkl
from itertools import compress
import time
import math
import argparse
from collections import OrderedDict, defaultdict
from misc.utilities import get_fname_from_path, Preload_embedding, load_vectors
from misc.torch_utility import load_model_states
from misc.data_loader import BatchDataLoader

import sys
from scipy.spatial.distance import cdist
from sklearn import metrics

import pdb

parser = argparse.ArgumentParser()
# Input data
parser.add_argument('--fp_test', default='./data/snli/snli_data.json')
parser.add_argument('--split', default='test')
#
parser.add_argument('--model_name', default='QAconcept', type=str)

#others
parser.add_argument('--best_model', default='./check_points/qaconcept_==easy==_9507775_B8_L1_H6_D1_May-10-2019-03:50PM__best.pt')
parser.add_argument('--loader_num_workers', type=int, default=4)
parser.add_argument('--batch_size',  type=int, default=32)

#printing information
parser.add_argument('--print_mode', default='json', choices=['json', 'simple'])


def Variable(data, *args, **kwargs):
    if USE_CUDA:
        return autograd.Variable(data.cuda(), *args, **kwargs)
    else:
        return autograd.Variable(data, *args, **kwargs)

def evaluate(data, params, use_mask = True, print_out = False):
    '''
        This function evaluates the model
    '''
    model.eval() # switch off the dropout if applied

    q1, a1, premise, hypothesis, label = data['q1'], data['q2'], data['qstr'], data['astr'], data['label']
    concept_qa, concept_aq = data['concept_qa'], data['concept_aq']
    q1, a1, label = Variable(q1), Variable(a1), Variable(label)
    concept_qa, concept_aq = Variable(concept_qa), Variable(concept_aq)

    mem_size = q1.size()[1]
    b_size = q1.size()[0]

    # feed data through the model
    if use_mask == True:
        qmask = Variable(data['qmask'], requires_grad=False) # qmask: [B, T]
        amask = Variable(data['amask'], requires_grad=False) # amask: [B, T]
        matching, q_attn_list, a_attn_list = model(q1, a1, qmask, amask, concept_qa, concept_aq, sharpening=params.sharpening,
                                concept_attention=params.concept_attention, alpha=params.alpha) # [B, 3]
    else:
        matching, q_attn_list, a_attn_list = model(q1, a1, concept_qa, concept_aq, sharpening=params.sharpening,
                                concept_attention=params.concept_attention, alpha=params.alpha) # [B, 3]

    # calculate word importance
    if args.model_name.lower() == 'semultitask':
        CE = nn.CrossEntropyLoss()
        LL = nn.BCELoss()

        concept_qa = concept_qa.permute(0,3,1,2) # [B, H, T1, T2]
        concept_aq = concept_aq.permute(0,3,1,2)

        loss_eval = (1. / (1. + params.multitask_scale)) * CE(matching.float(), label.long())
        for qa in q_attn_list:
            loss_eval = loss_eval + (params.multitask_scale / (1. + params.multitask_scale)) * LL(qa[:,:params.num_concepts,:,:].float(), concept_qa.float())
        for aq in a_attn_list:
            loss_eval = loss_eval + (params.multitask_scale / (1. + params.multitask_scale)) * LL(aq[:,:params.num_concepts,:,:].float(), concept_aq.float())
    else:
        criterion = nn.CrossEntropyLoss()
        loss_eval = criterion(matching.float(), label.long())

    # pdb.set_trace()

    matching = matching.data
    label = label.data

    ############################
    # predict relevancy
    ############################
    pred_score, pred_class = torch.max(matching.cpu(), 1) # [B], [B]

    label = torch.tensor(label, dtype=torch.int64).cpu() # [B]

    comp = (pred_class == label) # [B]
    correct  = comp.sum().data.item() # scalar
    precision = metrics.precision_score(label.numpy(), pred_class.numpy(), labels=[0,1,2], average=None)
    recall = metrics.recall_score(label.numpy(), pred_class.numpy(), labels=[0,1,2], average=None)
    f1 = metrics.f1_score(label.numpy(), pred_class.numpy(), labels=[0,1,2], average=None)

    # if correct < 8:
    #     print('predicted class is {}, label is {}'.format(pred_class, label))

    # print out some examples to console
    # if print_out:
    #     print("\rf1 is {}, precision is {}, recall is {}, accuracy is {}".format(f1, precision, recall, 1.0*correct/len(label)))
    #     print('\r-----------------------------------------------------------\n')

    return loss_eval.data.item(), correct, f1, precision, recall, b_size

# main
if __name__ == "__main__":

    ##############################
    #### Define (hyper)-parameters
    ##############################

    USE_CUDA =  torch.cuda.is_available()
    if USE_CUDA:
        print("We are using CUDA {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    else:
        print("No cuda detected")
    args = parser.parse_args()
    print(args.__dict__)

    ###############################
    # Load [best] model snapshot and Build model and optimizer
    ###############################
    fname_prefix = get_fname_from_path(args.best_model).split('.')[0]
    model_type = get_fname_from_path(args.best_model).split('_')[0]
    ###############################
    # Load [best] model snapshot
    ###############################
    state, params = load_model_states(args.best_model)
    torch.manual_seed(params.seed_random)

    if USE_CUDA:
        torch.cuda.manual_seed(params.seed_random)

    word_embd_dim =  params.fp_word_embd_dim

    concept_layers = [int(i) for i in params.concept_layers.split(',')]
    print("Adding external knowledge to layers {} ...\n".format(concept_layers))

    if args.model_name == 'QAcombine' or args.model_name == 'QAconcept':
        import models.QAcombine as net
        model = net.QAconcept(hidden_size = params.hidden_size, drop_rate = params.droprate,
                             num_layers = params.num_layers,
                             num_layers_cross = params.num_layers_cross,
                             heads = params.heads, embd_dim=params.fp_embd_dim,
                             word_embd_dim=params.fp_word_embd_dim,
                             num_concepts=params.num_concepts)
    elif args.model_name == 'KNLIresnet' or args.model_name == 'KNLIconceptResnet':
        import models.KNLIconceptResnet as net
        model = net.KNLIresnet(hidden_size = params.hidden_size, drop_rate = params.droprate,
                         num_layers = params.num_layers,
                         num_layers_cross = params.num_layers_cross,
                         heads = params.heads, embd_dim=params.fp_embd_dim,
                         word_embd_dim=params.fp_word_embd_dim,
                         num_concepts=params.num_concepts)

    elif args.model_name == 'QAse':
        import models.QAse as net
        model = net.QAse(hidden_size = params.hidden_size, drop_rate = params.droprate,
                         num_layers = params.num_layers,
                         num_layers_cross = params.num_layers_cross,
                         heads = params.heads, embd_dim=params.fp_embd_dim,
                         word_embd_dim=params.fp_word_embd_dim,
                         num_concepts=params.num_concepts)

    elif args.model_name == 'SEMH':
        import models.SEmultiHead as net
        model = net.SEMH(hidden_size = params.hidden_size, drop_rate = params.droprate,
                         num_layers = params.num_layers,
                         num_layers_cross = params.num_layers_cross,
                         heads = params.heads, embd_dim=params.fp_embd_dim,
                         word_embd_dim=params.fp_word_embd_dim,
                         num_concepts=params.num_concepts,
                         concept_layers=concept_layers)

    elif args.model_name.lower() == 'semultitask':
        import models.SEmultiTask as net
        model = net.SEMultitask(hidden_size = params.hidden_size, drop_rate = params.droprate,
                         num_layers = params.num_layers,
                         num_layers_cross = params.num_layers_cross,
                         heads = params.heads, embd_dim=params.fp_embd_dim,
                         word_embd_dim=params.fp_word_embd_dim,
                         num_concepts=params.num_concepts)

    print("loading the model %s...." % args.best_model)
    model.load_state_dict(state)

    ############################
    # add Masking flag
    ############################
    mask_data = True

    ############################
    # Load data
    ############################
    with open(params.concept_dict, 'rb') as f:
        concept_dict = pkl.load(f)

    if params.fp_embd.split('.')[-1] == 'bin':
        pre_embd = Preload_embedding(params.fp_embd, params.fp_embd_context, params.fp_embd_type)
    else:
        pre_embd = load_vectors(params.fp_embd)

    dset_test = BatchDataLoader(fpath = args.fp_test, embd_dict = pre_embd, concept_dict=concept_dict,
                                 split=args.split, emd_dim=params.fp_word_embd_dim, num_concepts = params.num_concepts)

    test_loader = data_utils.DataLoader(dset_test, batch_size = args.batch_size, shuffle=False, num_workers = args.loader_num_workers, drop_last=True)

    if USE_CUDA:
        model.cuda()

    total_params = 0
    for i in model.parameters():
        total_params += np.prod(i.size())

    print('-'*50)
    print(model)
    print("Total number of parameters is:", total_params)

    ############################
    # Start running the model
    ############################
    print('-'*50)
    print('input file:', args.fp_test)
    print()

    results = {}

    test_loss = 0.0
    test_correct = 0.0
    test_precision = np.zeros(3)
    test_recall = np.zeros(3)
    test_f1 = np.zeros(3)
    total_data = 0.0

    for j, sample in enumerate(test_loader, 0):
        print_out = False

        if j % 100 == 0:
            print_out = True

        test_stat = evaluate(sample, params, use_mask = mask_data, print_out = print_out)

        test_loss += test_stat[0]
        test_correct += test_stat[1]
        test_f1 += test_stat[2]
        test_precision += test_stat[3]
        test_recall += test_stat[4]
        total_data += test_stat[5]


    test_accuracy = test_correct / total_data
    test_f1 = test_f1 / len(test_loader)
    test_precision = test_precision / len(test_loader)
    test_recall = test_recall / len(test_loader)
    test_loss = test_loss / len(test_loader)

    results['accuracy'] = test_accuracy
    # results['f1'] = test_f1
    # results['precision'] = test_precision
    # results['recall'] = test_recall
    results['loss'] = test_loss

    print('the test loss is {}\n'.format(test_loss))
    print('the final accuracy is {}\n'.format(test_accuracy))
    print('the average F1 is {}\n'.format(test_f1))
    print('the average precision is {}\n'.format(test_precision))
    print('the average recall is {}\n'.format(test_recall))

    # save sim_results
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')
    with open('./outputs/'+fname_prefix+'_results.json', 'w') as out_file:
        print("Saving results ...\n")
        json.dump(results, out_file)
