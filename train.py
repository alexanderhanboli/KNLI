from __future__ import  print_function, division
import torch.utils.data as data_utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import os
import time
import math
import argparse
import json
from itertools import compress
from collections import OrderedDict
from misc.utilities import initialize_weights, timeSince, dump_to_json, create_dir, Preload_embedding, read_json_file, load_vectors
from misc.torch_utility import get_state, load_model_states
from misc.data_loader import BatchDataLoader

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
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--weight_decay', default=0.001, type=float)
parser.add_argument('--beta1', default=0.5, type=float)
parser.add_argument('--beta2', default=0.999, type=float)

# Model params
parser.add_argument('--droprate', type=float, default=0.1)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_layers_cross', type=int, default=4)
parser.add_argument('--heads', type=int, default=5)

#others
parser.add_argument('--loader_num_workers', type=int, default=5)
parser.add_argument('--print_every', type=int, default=200)
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
    optimizer.optimizer.zero_grad()
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
        matching, q_weights, q_cossim = model(q1, a1, qmask, amask, sharpening=args.sharpening, alpha=args.alpha) # [B, 3]
    else:
        matching, q_weights, q_cossim = model(q1, a1, sharpening=args.sharpening, alpha=args.alpha) # [B, 3]

    # calculate word importance
    criterion = nn.CrossEntropyLoss()
    loss_eval = criterion(matching.float(), label.long())

    q_weights = q_weights.data
    # q_cossim = q_cossim.data
    matching = matching.data
    label = label.data

    ############################
    # predict relevancy
    ############################
    pred_score, pred_class = torch.max(matching.cpu(), 1) # [B], [B]
    label = torch.tensor(label, dtype=torch.int64).cpu() # [B]
    comp = (pred_class == label) # [B]

    if comp.sum() > 0:
        correct_idx = comp.nonzero().squeeze(1).numpy()
        cflag = True
    else:
        cflag = False
    if (1-comp).sum() > 0:
        incorrect_idx = (1-comp).nonzero().squeeze(1).numpy()
        iflag = True
    else:
        iflag = False
    correct  = comp.sum().data.item() # scalar

    precision = metrics.precision_score(label.numpy(), pred_class.numpy(), average='macro')
    recall = metrics.recall_score(label.numpy(), pred_class.numpy(), average='macro')
    f1 = metrics.f1_score(label.numpy(), pred_class.numpy(), average='macro')

    # write incorrect examples to txt
    negative_examples = ''
    q_kept = (q_weights > 0).cpu().numpy().astype(int)
    q_weights = q_weights.cpu().numpy()
    # q_cossim = q_cossim.cpu().numpy()

    # print out some examples to console
    if print_out:
        # correct examples
        if cflag:
            ce = correct_idx[0]
            q_idx = list(q_kept[ce])
            q_word2match = list(compress(nltk.word_tokenize(premise[ce]), q_idx))
            print("\rCorrect:")
            print(
                "\rpremise: {}\nweights: {}\nwords to match: {}\nhypothesis: {}\nmatch score: {}, class: {}, pred_class: {}\n".format(
                premise[ce], q_weights[ce][q_weights[ce] > 0], q_word2match,
                hypothesis[ce],
                                    pred_score.cpu().numpy()[ce],
                                    label.numpy()[ce],
                                    pred_class.numpy()[ce]))

        # incorrect examples
        if iflag:
            ie = incorrect_idx[0]
            q_idx = list(q_kept[ie])
            q_word2match = list(compress(nltk.word_tokenize(premise[ie]), q_idx))
            print("\rIncorrect:")
            print(
                "\rpremise: {}\nweights: {}\nwords to match: {}\nhypothesis: {}\nmatch score: {}, class: {}, pred_class: {}\n".format(
                premise[ie], q_weights[ie][q_weights[ie] > 0], q_word2match,
                hypothesis[ie],
                                    pred_score.cpu().numpy()[ie],
                                    label.numpy()[ie],
                                    pred_class.numpy()[ie]))

        print("\rf1 is {}, precision is {}, recall is {}, accuracy is {}".format(f1, precision, recall, 1.0*correct/len(label)))
        print('\r-----------------------------------------------------------\n')

    # not critical, but just in case
    del q1, a1, premise, hypothesis, label

    return loss_eval.data.item(), correct, f1, precision, recall, negative_examples


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
        elif args.fp_embd.split('.')[-1] == 'vec' or args.fp_embd.split('.')[-1] == 'txt':
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

    # initialize weights as the same in Transformer paper: Glorot / fan_avg
    print("Initializing weights ...")
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

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
    optimizer_adam = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           betas=(args.beta1, args.beta2), eps=1e-9, weight_decay = args.weight_decay)
    optimizer = net.NoamOpt(args.fp_embd_dim, 1.8, 5000, optimizer_adam)
    # optimizer = net.AdamDecay(args.learning_rate, 0.9**(1.0/10000), optimizer_adam)

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
    negative_examples = ''
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
        print('Current training rate {}\n'.format(optimizer._rate))
        print('Train EPC_loss %.4f\n' % (epc_loss))

        if epoch % args.val_interval == 0:

            val_loss = 0.0
            val_correct = 0.0
            val_f1 = 0.0
            val_precision = 0.0
            val_recall = 0.0
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
                negative_examples += val_loss_correct[5]

            val_f1 = val_f1 / len(val_loader)
            val_precision = val_precision / len(val_loader)
            val_recall = val_recall / len(val_loader)
            val_correct = val_correct / (len(val_loader)*args.batch_size)
            stats['val_losses'].append(val_loss)
            stats['val_losses_ts'].append(epoch)
            stats['val_f1'].append(val_f1)
            stats['val_precision'].append(val_precision)
            stats['val_recall'].append(val_recall)
            stats['val_acc'].append(val_correct)

        if val_f1 > best_val_f1:
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

    # file_dir = '/home/ec2-user/outputs'
    # if not os.path.exists(file_dir):
    #     os.makedirs(file_dir)
    # with open("/home/ec2-user/outputs/negative_examples.txt", "w+") as text_file:
    #     print("Printing negative examples...\n")
    #     print(negative_examples, file=text_file)
