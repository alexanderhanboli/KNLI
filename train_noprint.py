from __future__ import  print_function, division
import torch.utils.data as data_utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import os, io
import time
import math
import argparse
import json
from itertools import compress
from collections import OrderedDict
from misc.utilities import initialize_weights, timeSince, dump_to_json, create_dir, Preload_embedding, read_json_file
from misc.torch_utility import get_state, load_model_states
from misc.data_loader import BatchDataLoader
from misc.opt import OpenAIAdam

import sys
from sklearn import metrics
import nltk
nltk.download('punkt')

parser = argparse.ArgumentParser()
# Input data
parser.add_argument('--fp_train', default='./data/snli_data.json')
parser.add_argument('--fp_val',   default='./data/snli_data.json')
parser.add_argument('--fp_embd',  default='./data/wiki.en.bin')
parser.add_argument('--fp_word_embd_dim',  default=300, type=int)
parser.add_argument('--fp_embd_dim',  default=300, type=int)
parser.add_argument('--fp_embd_context',  default='')
parser.add_argument('--fp_embd_type',  default='generic')
parser.add_argument('--bert_embd', default=False, action='store_true')
parser.add_argument('--bert_layers', default=10, type=int)

# Module optim options
parser.add_argument('--opt', default='openai', type=str, choices=['original', 'openai'])
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--lr', type=float, default=6.25e-5)
parser.add_argument('--lr_warmup', type=float, default=0.002)
parser.add_argument('--l2', type=float, default=0.01)
parser.add_argument('--vector_l2', action='store_true')
parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
parser.add_argument('--e', type=float, default=1e-8)
parser.add_argument('--max_grad_norm', type=int, default=1)

# Model params
parser.add_argument('--droprate', type=float, default=0.1)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_layers_cross', type=int, default=1)
parser.add_argument('--heads', type=int, default=5)

#others
parser.add_argument('--loader_num_workers', type=int, default=5)
parser.add_argument('--print_every', type=int, default=1000)
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--n_epochs', default=50, type=int)
parser.add_argument('--check_point_dir', default='./check_points/')
parser.add_argument('--log_id', default='dummy123')
parser.add_argument('--checkpoint_every', default=10, type=int)
parser.add_argument('--seed_random', default=1019, type=int)
parser.add_argument('--cudnn_enabled', default=1, type=int)
parser.add_argument('--model_name', default='QAsim')
parser.add_argument('--load_model', default=False, action='store_true')

#
# parser.add_argument('--rule_based', default=False, type=bool)
parser.add_argument('--sharpening', default=False, action='store_true')
parser.add_argument('--weight_thrd', default=0.00, type=float) # threshold for selection
# parser.add_argument('--sim_thrd', default=0.80, type=float) # threshold for similarity
parser.add_argument('--neg_sampling_ratio', default=1, type=float) # sampling ration: 1 means 50-50, 1< means less negative example, 0 means no negative example
parser.add_argument('--alpha', default=0.00, type=float)

#
parser.add_argument('--debug', default=False, action='store_true')

# help functions
def Variable(data, *args, **kwargs):

    if USE_CUDA:
        return autograd.Variable(data.cuda(), *args, **kwargs)
    else:
        return autograd.Variable(data, *args, **kwargs)

def createFileName(args):

    # create folder if not there
    create_dir(args.check_point_dir)

    return os.path.join(args.check_point_dir, str.lower(args.model_name) + "_" + args.log_id +
                        '_B' + str(args.batch_size) + '_L' + str(args.num_layers) +
                        '_H' + str(args.heads) + '_D' + str(args.droprate-int(args.droprate))[1:][1])

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data

# train step
def train(data, use_mask = True):
    '''
        This function runs one step of training loop
        [question 1, question2, label]
        label: [B]
    '''
    model.train()
    q1, a1, label = data['q1'], data['q2'], data['label']
    q1, a1, label = Variable(q1), Variable(a1), Variable(label)

    # setup the optim
    if args.opt == 'original':
        optimizer.optimizer.zero_grad()
    elif args.opt == 'openai':
        optimizer.zero_grad()
    loss = 0.0

    # feed data through the model
    if use_mask == True:
        qmask = Variable(data['qmask'], requires_grad=False) # qmask: [B, T]
        amask = Variable(data['amask'], requires_grad=False) # amask: [B, T]
        matching, _, _ = model(q1, a1, qmask, amask, sharpening=args.sharpening, alpha=args.alpha) # [B, 3]
    else:
        matching, _, _ = model(q1, a1, sharpening=args.sharpening, alpha=args.alpha) # [B, 3]

    # calculate loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(matching.float(), label.long())

    # do backprop and udpate
    loss.backward()
    optimizer.step()

    # not critical, but just in case
    del q1, a1, label

    return loss.data.item()

# evaluate step
def evaluate(data, use_mask = True, print_out = False):
    '''
        This function evaluates the model
    '''
    model.eval() # switch off the dropout if applied

    q1, a1, premise, hypothesis, label = data['q1'], data['q2'], data['qstr'], data['astr'], data['label']
    q1, a1, label = Variable(q1), Variable(a1), Variable(label)

    mem_size = q1.size()[1]
    b_size = q1.size()[0]

    # feed data through the model
    if use_mask == True:
        qmask = Variable(data['qmask'], requires_grad=False) # qmask: [B, T]
        amask = Variable(data['amask'], requires_grad=False) # amask: [B, T]
        matching, _, _ = model(q1, a1, qmask, amask, sharpening=args.sharpening, alpha=args.alpha) # [B, 3]
    else:
        matching, _, _ = model(q1, a1, sharpening=args.sharpening, alpha=args.alpha) # [B, 3]

    # calculate word importance
    criterion = nn.CrossEntropyLoss()
    loss_eval = criterion(matching.float(), label.long())

    matching = matching.data
    label = label.data

    ############################
    # predict relevancy
    ############################
    pred_score, pred_class = torch.max(matching.cpu(), 1) # [B], [B]
    label = torch.tensor(label, dtype=torch.int64).cpu() # [B]

    comp = (pred_class == label) # [B]
    correct  = comp.sum().data.item() # scalar
    precision = metrics.precision_score(label.numpy(), pred_class.numpy(), average='macro')
    recall = metrics.recall_score(label.numpy(), pred_class.numpy(), average='macro')
    f1 = metrics.f1_score(label.numpy(), pred_class.numpy(), average='macro')

    # print out some examples to console
    if print_out:
        print("\rf1 is {}, precision is {}, recall is {}, accuracy is {}".format(f1, precision, recall, 1.0*correct/len(label)))
        print('\r-----------------------------------------------------------\n')

    # not critical, but just in case
    del q1, a1, premise, hypothesis, label

    return loss_eval.data.item(), correct, f1, precision, recall, b_size


if __name__ == "__main__":

    ##############################
    #### Define (hyper)-parameters
    ##############################
    args = parser.parse_args()

    if args.bert_embd:
        print("Using BERT embeddings...")
        assert args.fp_word_embd_dim == 768
    else:
        print("Using fastext embeddings...")
        assert args.fp_word_embd_dim == 300

    if args.cudnn_enabled == 1:
        torch.backends.cudnn.enabled = True

    USE_CUDA = torch.cuda.is_available()

    if USE_CUDA:
        print("Yayy we use CUDA {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    else:
        print("No cuda detected")

    torch.manual_seed(args.seed_random)
    np.random.seed(args.seed_random)

    if USE_CUDA:
        torch.cuda.manual_seed(args.seed_random)

    print(args.__dict__)
    fname_part = createFileName(args) # just creat a generic file name

    ############################
    # Load data
    ############################
    enable_sampler = False

    if args.bert_embd:
        dset_train = BatchDataLoaderBert(fpath = args.fp_train, split='train',
                                    emd_dim=args.fp_word_embd_dim, num_bert_layers=args.bert_layers)

        dset_val   = BatchDataLoaderBert(fpath = args.fp_val, split='val',
                                    emd_dim=args.fp_word_embd_dim, num_bert_layers=args.bert_layers)
    else:
        if args.fp_embd.split('.')[-1] == 'bin':
            pre_embd = Preload_embedding(args.fp_embd, args.fp_embd_context, args.fp_embd_type)
        else:
            pre_embd = load_vectors(args.fp_embd)

        dset_train = BatchDataLoader(fpath = args.fp_train, embd_dict = pre_embd,
                                     split='train', emd_dim=args.fp_word_embd_dim)

        dset_val   = BatchDataLoader(fpath = args.fp_val, embd_dict = pre_embd,
                                    split='dev', emd_dim=args.fp_word_embd_dim)

    if enable_sampler == True:
        sampler = defineSampler(args.fp_train, neg_sampling_ratio = args.neg_sampling_ratio) # helps with imbalanced classes
        train_loader = data_utils.DataLoader(dset_train, batch_size=args.batch_size,
            shuffle=False, num_workers=args.loader_num_workers, sampler=sampler, drop_last=True)
    else:
        train_loader = data_utils.DataLoader(dset_train, batch_size=args.batch_size,
            shuffle=True, num_workers=args.loader_num_workers, drop_last=True)

    val_loader = data_utils.DataLoader(dset_val, batch_size=args.batch_size, shuffle=True,
            num_workers=5, drop_last=True)

    ############################
    # add Masking flag
    ############################
    mask_data = True

    ############################
    # Build model and optimizer
    ############################
    if args.model_name == 'Qkeywords':
        import models.Qkeywords as net
        model = net.Qkeywords(hidden_size = args.hidden_size, drop_rate = args.droprate,
                         num_layers = args.num_layers,
                         heads = args.heads, embd_dim=args.fp_embd_dim,
                         word_embd_dim=args.fp_word_embd_dim)

    elif args.model_name == 'QattendA':
        import models.QattendA as net
        model = net.QatA(hidden_size = args.hidden_size, drop_rate = args.droprate,
                         num_layers = args.num_layers,
                         num_layers_cross = args.num_layers_cross,
                         heads = args.heads, embd_dim=args.fp_embd_dim,
                         word_embd_dim=args.fp_word_embd_dim)

    elif args.model_name == 'QAembd':
        import models.QAembd as net
        model = net.QAembd(hidden_size = args.hidden_size, drop_rate = args.droprate,
                         num_layers = args.num_layers,
                         num_layers_cross = args.num_layers_cross,
                         heads = args.heads, embd_dim=args.fp_embd_dim,
                         word_embd_dim=args.fp_word_embd_dim)

    elif args.model_name == 'QAsim':
        import models.QAsim as net
        model = net.QAsim(hidden_size = args.hidden_size, drop_rate = args.droprate,
                         num_layers = args.num_layers,
                         num_layers_cross = args.num_layers_cross,
                         heads = args.heads, embd_dim=args.fp_embd_dim,
                         word_embd_dim=args.fp_word_embd_dim)

    elif args.model_name == 'QAesim':
        import models.QAesim as net
        model = net.QAesim(hidden_size = args.hidden_size, drop_rate = args.droprate,
                         num_layers = args.num_layers,
                         num_layers_cross = args.num_layers_cross,
                         heads = args.heads, embd_dim=args.fp_embd_dim,
                         word_embd_dim=args.fp_word_embd_dim)

    elif args.model_name == 'QAcombine':
        import models.QAcombine as net
        model = net.QAcombine(hidden_size = args.hidden_size, drop_rate = args.droprate,
                         num_layers = args.num_layers,
                         num_layers_cross = args.num_layers_cross,
                         heads = args.heads, embd_dim=args.fp_embd_dim,
                         word_embd_dim=args.fp_word_embd_dim)

    #########################
    # Check whether there is
    # snapshot of current model
    ########################
    best_load_ext = ''
    if args.load_model == True and os.path.exists(fname_part + '.pt') == True:
        # since we loading the same thing no need to upload
        # parameters
        print("Loading %s" % fname_part + '.pt')
        state, _ = load_model_states(fname_part + '.pt')
        model.load_state_dict(state)
        best_load_ext = "_LO"
        del state

    #########################
    # Add loss function and other configs
    ########################
    if args.opt == 'original':
        optimizer_adam = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               betas=(args.beta1, args.beta2), eps=args.e, weight_decay = args.l2)
        optimizer = net.NoamOpt(args.fp_embd_dim, 2, 5000, optimizer_adam)
        # optimizer = net.AdamDecay(args.learning_rate, 0.9**(1.0/10000), optimizer_adam)
    elif args.opt == 'openai':
        n_updates_total = len(train_loader) * args.n_epochs
        optimizer = OpenAIAdam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr,
                               schedule=args.lr_schedule,
                               warmup=args.lr_warmup,
                               t_total=n_updates_total,
                               b1=args.beta1,
                               b2=args.beta2,
                               e=args.e,
                               l2=args.l2,
                               vector_l2=args.vector_l2,
                               max_grad_norm=args.max_grad_norm)

    # if there is GPU, make them cuda
    if USE_CUDA:
        model.cuda()

    total_params = 0
    total_params_Trainable = 0

    for i in model.parameters():
        total_params += np.prod(i.size())
        if (i.requires_grad == True):
            total_params_Trainable += np.prod(i.size())

    print(model)
    print("Total number of ALL parameters: %d" % total_params)
    print("Total number of TRAINABLE parameters: %d" % total_params_Trainable)

    # if args.rule_based:
    #     print("Evaluating using rule based modifications!")

    ############################
    # Start running the model
    ############################
    start = time.time()
    best_val = np.inf
    best_val_acc = -np.inf
    best_val_f1  = -np.inf
    best_val_precision = -np.inf
    best_val_recall = -np.inf
    best_epoch = 0
    mem_loader = False
    print_phrase = 'Error'

    stats = {
            'train_losses': [], 'train_losses_ts':[],'train_losses_epc':[],
            'val_losses': [],  'val_losses_ts':[], 'val_acc':[],
            'val_f1':[],'val_precision':[], 'val_recall':[],
            'best_val_loss': -1, 'best_ts':0, 'best_val_accuracy': -1, 'best_val_f1': -1,
            'best_precision':-1, 'best_recall':-1
            }
    time_step = 0
    for epoch in range(1, args.n_epochs + 1):

        progress = 100 * epoch / args.n_epochs
        running_loss = 0.0
        epc_loss = 0.0

        for i, data_sample in enumerate(train_loader, 0):

            time_step += 1
            batch_loss = train(data_sample, use_mask = mask_data)
            running_loss += batch_loss
            epc_loss += batch_loss

            if (i + 1) % args.print_every == 0:

                stats['train_losses'].append(running_loss)
                stats['train_losses_ts'].append(time_step)
                loss = running_loss / args.print_every
                print('Time %s, Epcoh %d, Progress/Sample(%d%%), %s %.4f' % (timeSince(start),
                                        epoch, i / len(train_loader) * 100, print_phrase, loss))
                running_loss = 0

                if args.debug:
                    print('debug mode...\n')
                    break

        epc_loss = epc_loss/len(train_loader)
        stats['train_losses_epc'].append(epc_loss)
        print('Train EPC_loss %.4f\n' % (epc_loss))

        if epoch % args.val_interval == 0:

            val_loss = 0.0
            val_correct = 0.0
            val_f1 = 0.0
            val_precision = 0.0
            val_recall = 0.0
            total_data = 0
            for j, data_val in enumerate(val_loader, 0):

                print_out = False
                if j % 10 == 0:
                    print_out = True
                val_loss_correct = evaluate(data_val, use_mask = mask_data, print_out = print_out)
                val_loss    += val_loss_correct[0]
                val_correct += val_loss_correct[1]
                val_f1 += val_loss_correct[2]
                val_precision += val_loss_correct[3]
                val_recall +=val_loss_correct[4]
                total_data += val_loss_correct[5]

            val_f1 = val_f1 / len(val_loader)
            val_precision = val_precision / len(val_loader)
            val_recall = val_recall / len(val_loader)
            val_correct = val_correct / total_data
            stats['val_losses'].append(val_loss)
            stats['val_losses_ts'].append(epoch)
            stats['val_f1'].append(val_f1)
            stats['val_precision'].append(val_precision)
            stats['val_recall'].append(val_recall)
            stats['val_acc'].append(val_correct)

        if val_correct > best_val_acc:
                best_val_f1 = val_f1
                best_val_acc = val_correct
                best_val_precision = val_precision
                best_val_recall = val_recall
                best_epoch = epoch

                ###############
                # Save best model in pt file and other data in json file
                ###############
                fname_ck =  fname_part + best_load_ext +'_best.pt'
                fname_json =  fname_part + best_load_ext + '_best.json'
                stats['best_val_accuracy'] = best_val_acc
                stats['best_val_f1'] = best_val_f1
                stats['best_precision'] = best_val_precision
                stats['best_recall'] = best_val_recall
                stats['best_ts'] = (epoch, time_step)
                best_state = get_state(model)
                print('Saving best model so far in epoch %d to %s' % (epoch, fname_ck))
                checkpoint = {
                                'args': args.__dict__,
                                'model_states': best_state,
                             }
                for k, v in stats.items():
                    checkpoint[k] = v

                torch.save(checkpoint, fname_ck)

                del checkpoint['model_states']
                del best_state

                dump_to_json(fname_json, checkpoint)
                print("Best val f1 %.4f, precision %.4f, recall %.4f, and acc %.4f so far in epoch %d after %d steps"
                       % (best_val_f1, best_val_precision, best_val_recall, best_val_acc, epoch, optimizer._step))

        else:
                print('Validation f1 %.4f, precision %.4f, recall %.4f, and accuracy %.4f in epoch %d after %d steps'
                       % (val_f1, val_precision, val_recall, val_correct, epoch, optimizer._step))
                print("Best val f1 %.4f, precision %.4f, recall %.4f, and acc %.4f so far in epoch %d"
                       % (best_val_f1, best_val_precision, best_val_recall, best_val_acc, best_epoch))


        if epoch % args.checkpoint_every == 0:

                ###############
                # Save check model in pt file and other data in json file
                ###############
                fname_ck =  fname_part + '.pt'
                fname_json =  fname_part + '.json'
                curr_state = get_state(model)
                print('\r**********Saving checkpoint for in epoch {} to {}**********'.format(epoch, fname_ck))
                checkpoint = {
                                'args': args.__dict__,
                                'model_states': curr_state,
                             }
                for k, v in stats.items():
                    checkpoint[k] = v

                torch.save(checkpoint, fname_ck)
                del checkpoint['model_states']
                del curr_state

                dump_to_json(fname_json, checkpoint)

    if USE_CUDA:
        torch.cuda.empty_cache()
