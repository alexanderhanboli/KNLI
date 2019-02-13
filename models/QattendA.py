from __future__ import  print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import math, copy, time
from torch.autograd import Variable

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, decay = 0.5):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._decay = decay

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-self._decay) *
            min(step ** (-self._decay), step * self.warmup ** (-1-self._decay)))

class AdamDecay:
    "Optim wrapper that implements rate."
    def __init__(self, lr, decay, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self._rate = lr
        self._decay = decay

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self._rate * self._decay ** step)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    '''
     Compute Scale Dot product
     query, key, value are B*T*D
     B is the batch size
     T is the sentence length
     D is the dimension of each word
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k) # [B, H, T1, T2]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1) # [B, H, T1, T2]

    # apply dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, dim, eps=1e-6):

        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        '''
           This module applies layer norm
        '''
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        norm = self.alpha * (x - mean) / (std + self.eps) + self.beta

        return norm

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model = 300, d_ff = 512, dropout=0.1):

        super(PositionwiseFeedForward, self).__init__()

        ### define the linear layer
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
            x is B*T*D
        '''
        out = self.ff2(self.dropout(F.relu(self.ff1(x))))

        return out

class MultiHeadedAttn(nn.Module):

    def __init__(self, heads=4, d_model=300, dropout=0.1):
        "Take in number of heads, model size, and dropout rate."
        super(MultiHeadedAttn, self).__init__()
        assert d_model % heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // heads
        self.heads = heads

        ## Linear layers for multi-head attention
        self.fwQ  = nn.Linear(d_model, d_model) # for the query
        self.fwK  = nn.Linear(d_model, d_model) # for the key
        self.fwV  = nn.Linear(d_model, d_model) # for the answer
        # W0
        self.fwMH = nn.Linear(d_model, d_model) # output linear layer

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
            query, key, value are B*T*D
            B: is batch size
            T: is sequence length
            D: is model dimension
        '''
        if mask is not None:
            mask = mask.unsqueeze(1) # [B, 1, T]
            mask = mask.unsqueeze(1) # [B, 1, 1, T]

        batch_size = query.size(0)

        ##
        # Step 1: apply the linear layer
        ##
        query = self.fwQ(query) # [B, T, D]
        key   = self.fwK(key)
        value = self.fwV(value)

        ##
        # Step 2: change the shape to d_k, since we pply attention per d_k not d_model
        # now the shape of query is batch_size * heads * sentence_length(T) * d_k
        ##
        query = query.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2) # [B, T, H, dk]
        key   = key.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)

        ##
        # Step 3) do attention
        # query, key, value are all of the shape [B, heads, T, d_k]
        ##
        x, self.attn  = attention(query, key, value, mask=mask, dropout=self.dropout)

        ##
        # Step 4.1) "Concat" using a view
        # output is of shape [batch_size, sentence_length, d_model]
        ##
        x = x.transpose(1, 2).contiguous() \
             .view(batch_size, -1, self.heads * self.d_k)
        ##
        # Step 4) get the output by apply a liner layer
        ##
        out = self.fwMH(x) # [batch_size, sentence_length, d_model]

        return out

class SimAttn(nn.Module):

    def __init__(self, heads=1, d_model=300, dropout=0.0):
        "Take in number of heads, model size, and dropout rate."
        super(SimAttn, self).__init__()
        assert d_model % heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // heads
        self.heads = heads

        ## Linear layers for multi-head attention
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, sharpening=True, mask=None):
        '''
            query, key, value are B*T*D
            B: is batch size
            T: is sequence length
            D: is model dimension
        '''
        if mask is not None:
            mask = mask.unsqueeze(1) # [B, 1, T]
            mask = mask.unsqueeze(1) # [B, 1, 1, T]

        batch_size = query.size(0)

        ##
        # Step 2: change the shape to d_k, since we pply attention per d_k not d_model
        # now the shape of query is batch_size * heads * sentence_length(T) * d_k
        ##
        query = query.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2) # [B, T, H, dk]
        key   = key.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)

        ##
        # Step 3) do attention
        # query, key, value are all of the shape [B, heads, T, d_k]
        ##
        x, self.attn  = attention(query, key, value, mask=mask, dropout=self.dropout)

        # sharpening the attention distribution
        if sharpening:
            self.attn = F.softmax(1000 * self.attn, dim=-1).clamp(0, 1)
            x = torch.matmul(self.attn, value)

        ##
        # Step 4.1) "Concat" using a view
        # output is of shape [batch_size, sentence_length, d_model]
        ##
        out = x.transpose(1, 2).contiguous() \
             .view(batch_size, -1, self.heads * self.d_k) # [B, T, D]

        return out

class Highway(nn.Module):
    def __init__(self, h_size, h_out, num_layers=2):

        super(Highway, self).__init__()
        self.num_layers = num_layers

        self.linear = nn.ModuleList([nn.Linear(h_size, h_size) for _ in range(self.num_layers)])
        self.gate = nn.ModuleList([nn.Linear(h_size, h_size) for _ in range(self.num_layers)])
        self.fc = nn.Linear(h_size, h_out)

    def forward(self, x):
        '''
            Input x is B*T*D
            y = H(x,WH)· T(x,WT) + x · (1 − T(x,WT)).
        '''
        for i in range(self.num_layers):

            H = F.relu(self.linear[i](x))
            T = F.sigmoid(self.gate[i](x))

            x = T * H + (1 - T) * x

        return self.fc(x)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N=4):

        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):

        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, mask=None):

        # self attention layer w/ resnet
        res = self.norm(x)
        res = self.self_attn(res, res, res, mask)
        res = self.dropout(res)
        x = x + res

        # feed forward layer w/ resnet
        res = self.norm(x)
        res = self.feed_forward(res)
        res = self.dropout(res)
        x = x + res

        return x

class CrEncoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N=4):

        super(CrEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, m, qmask=None, amask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, m, qmask, amask)

        return self.norm(x)

class CrEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):

        super(CrEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, m, qmask=None, amask=None):

        # self attention layer w/ resnet
        res = self.norm(x)
        res = self.self_attn(res, res, res, qmask)
        res = self.dropout(res)
        x = x + res

        # cross attention
        res = self.norm(x)
        res = self.cross_attn(res, m, m, amask)
        res = self.dropout(res)
        x = x + res

        # feed forward layer w/ resnet
        res = self.norm(x)
        res = self.feed_forward(res)
        res = self.dropout(res)
        x = x + res

        return x

class Sim(nn.Module):
    def __init__(self, embd_dim, hidden_size):

        super(Sim, self).__init__()

        self.cos = nn.CosineSimilarity(dim=-1)

        self.simf = nn.Sequential(
            nn.Linear(embd_dim * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x, y, alpha=0):
        '''
        x: [B, T, D]
        '''
        s0 = self.cos(x, y).unsqueeze(2).repeat(1, 1, 3) # [B, T, 3]

        input = torch.cat((x, y, x-y, x.mul(y)), -1) # [B, T, 4*D]
        s1 = self.simf(input) # [B, T, 3]

        score = alpha*s0 + (1-alpha)*s1

        return score

##############################
#### STEM model
##############################
class QatA(nn.Module):

    def __init__(self, hidden_size = 512, drop_rate = 0.1,
                 num_layers = 4, num_layers_cross = 2, heads = 4, embd_dim = 300, word_embd_dim = 300):
        '''
            Simple baseline model:
            This model use a simple summation of Fasttext word embeddings to represent each question in the pair.
            Need to make sure that embd_dim % heads == 0.
        '''
        super(QatA, self).__init__()
        c = copy.deepcopy
        if word_embd_dim == None:
            word_embd_dim = embd_dim
        self.word_embd_dim = word_embd_dim
        self.embd_dim = embd_dim
        self.drop = nn.Dropout(drop_rate)
        self.num_layers = num_layers
        self.heads = heads
        self.hidden_size = hidden_size

        # layers
        self.highway = Highway(h_size=word_embd_dim, h_out=embd_dim, num_layers=2)

        attn = MultiHeadedAttn(heads=heads, d_model=embd_dim, dropout=drop_rate)

        ff = PositionwiseFeedForward(d_model=embd_dim, d_ff=hidden_size, dropout=drop_rate)

        self.position = PositionalEncoding(d_model=embd_dim, dropout=drop_rate)

        self.encoder_q = Encoder(layer = EncoderLayer(size=embd_dim, self_attn=c(attn),
                                                feed_forward=c(ff), dropout=drop_rate), N=num_layers)

        # self.encoder_a = Encoder(layer = EncoderLayer(size=embd_dim, self_attn=c(attn),
        #                                         feed_forward=c(ff), dropout=drop_rate), N=num_layers)

        self.encoder_qa = CrEncoder(layer = CrEncoderLayer(size=embd_dim, self_attn=c(attn), cross_attn=c(attn),
                                                feed_forward=c(ff), dropout=drop_rate), N=num_layers_cross)

        self.coattention = SimAttn(heads=1, d_model=embd_dim, dropout=0.0)

        self.fc_q = nn.Sequential(
            nn.Linear(embd_dim, hidden_size),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, 1),
        )

        self.sim = Sim(embd_dim=embd_dim, hidden_size=hidden_size)

    def forward(self, question, answer, qmask=None, amask=None, sharpening=True, thrd=0.10, alpha=0.00):
        '''
         input : batch, seq_len
                qxlen [qx, lenx]
                qX: question one or two
                question: BxTxD
                memory:   BxTxD
                w1, w2: [B,T,D]
                qlength: [B, 1]
        '''

        ####
        # Step: apply highway and/or drop to the inputs and positional encoding
        ####
        # question = self.drop(self.highway(question))
        question = self.highway(question)
        answer = self.highway(answer) # share weights

        ####
        # Step: Coattention with each other and self attention using the same weights
        ####
        question_align = self.coattention(question, answer, answer, sharpening, amask) # [B, T, D]
        q_cossim = self.sim(question, question_align, alpha)# [B, T, 3]

        ####
        # Step: Self attention question and memory
        ####
        question = self.position(question)
        answer = self.position(answer) # add positional encoding

        #residual connection + layer norm
        question = self.encoder_q(question, qmask)
        answer = self.encoder_q(answer, amask)

        # cross attention
        question = self.encoder_qa(question, answer, qmask, amask) # question: [B, T, D], mask: [B, T]

        ###
        # Step: Final linear layer
        ####
        q_uweight = self.fc_q(question) # [B, T, 1]

        if qmask is not None:
            q_weight = F.softmax(q_uweight.masked_fill(qmask.unsqueeze(2) == 0, -1e9), dim=1) # [B, T, 1]
        else:
            q_weight = F.softmax(q_uweight, dim=1) # [B, T, 1]

        score = torch.sum(q_weight.mul(q_cossim), dim=1).clamp(0, 1).squeeze(1) # [B, 3]

        return score, q_weight.squeeze(), q_cossim
