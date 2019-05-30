from __future__ import  print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import math, copy, time
from torch.autograd import Variable

import pdb

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

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

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

"""LayerNorm
"""
# try:
#     from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
# except ImportError:
# print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.variance_epsilon)
        return self.weight * x + self.bias

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
        out = self.ff2(self.dropout(gelu(self.ff1(x))))

        return out


class MHSEAttn(nn.Module):

    def __init__(self, heads=5, d_model=300, dropout=0.1):
        "Take in number of heads, model size, and dropout rate."
        super(MHSEAttn, self).__init__()
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

    def forward(self, qa_concept, query, key, value, lambd=5.0, mask=None):
        '''
            query, key, value are B*T*D
            B: is batch size
            T: is sequence length
            D: is model dimension
            qa_concept: [B, T1, T2, H]
        '''
        if mask is not None:
            mask = mask.unsqueeze(1) # [B, 1, T]
            mask = mask.unsqueeze(1) # [B, 1, 1, T]

        batch_size = value.size(0)

        # Structured embeddings
        query = self.fwQ(query) # [B, T, D]
        key   = self.fwK(key)
        value = self.fwV(value)

        # Reshape
        query = query.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2) # [B, T, H, dk]
        key   = key.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2) # [B, H, T2, d_k]

        # Calculate initial scores
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(self.d_k) # [B, H, T1, T2]

        # Modify scores using external knowledge
        if qa_concept is not None:
            num_concepts = qa_concept.shape[3]
            scores[:,:num_concepts,:,:] = scores[:,:num_concepts,:,:] + lambd * qa_concept.permute(0, 3, 1, 2) # [B, H, T1, T2]

        # Add mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Calculate attention by applying softmax
        self.attn = self.dropout(F.softmax(scores, dim = -1)) # [B, H, T1, T2]

        x = torch.matmul(self.attn, value) # [B, H, T1, d_k]

        x = x.transpose(1, 2).contiguous() \
             .view(batch_size, -1, self.heads * self.d_k)

        out = self.fwMH(x) # [batch_size, sentence_length, d_model]

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

            H = gelu(self.linear[i](x))
            T = F.sigmoid(self.gate[i](x))

            x = T * H + (1 - T) * x

        return self.fc(x)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(d_model, eps=1e-12)
        # Compute the positional encodings once in log space.
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_length = x.size(1) # T
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), x.size(1)) # [B, T]
        position_embeddings = self.position_embeddings(position_ids) # [B, T, D]
        x = x + position_embeddings
        x = self.norm(x)
        return self.dropout(x)

class SegmentEncoding(nn.Module):
    "Implement the SE function."
    def __init__(self, d_model, dropout, max_len=4):
        super(SegmentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(d_model, eps=1e-12)
        # Compute the positional encodings once in log space.
        self.segment_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x, segment_ids):
        # segment_ids [B, T]
        segment_embeddings = self.segment_embeddings(segment_ids) # [B, T, D]
        x = x + segment_embeddings
        x = self.norm(x)
        return self.dropout(x)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N=4, concept_layers=[-1]):

        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size, eps=1e-12)
        self.concept_layers = concept_layers

    def forward(self, x, mask=None, concept=None, lambd=5.0):
        "Pass the input (and mask) through each layer in turn."
        for i, layer in enumerate(self.layers):
            if i in self.concept_layers:
                x = layer(x, mask, concept, lambd)
            else:
                x = layer(x, mask, None, 0.0)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):

        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm_in = LayerNorm(size, eps=1e-12)
        self.norm_out = LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, mask=None, concept=None, lambd=5.0):
        # self attention layer w/ resnet
        res = self.norm_in(x)
        res = self.self_attn(concept, res, res, res, lambd, mask)
        res = self.dropout(res)
        x = x + res

        # feed forward layer w/ resnet
        res = self.norm_out(x)
        res = self.feed_forward(res)
        res = self.dropout(res)
        x = x + res

        return x

class Classifier(nn.Module):
    def __init__(self, hidden_size, dropout):

        super(Classifier, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.simf = nn.Linear(hidden_size, 3)

    def forward(self, input):
        '''
        x: [B, T, D]
        '''
        score = self.simf(self.dropout(input)) # [B, 3]

        return score

##############################
#### QAse model
##############################
class BERTse(nn.Module):

    def __init__(self, hidden_size = 512, drop_rate = 0.1,
                 num_layers = 4, heads = 4,
                 embd_dim = 300, word_embd_dim = 300,
                 num_concepts = 5,
                 concept_layers = [-1]):
        super(BERTse, self).__init__()
        c = copy.deepcopy

        if word_embd_dim == None:
            word_embd_dim = embd_dim

        self.word_embd_dim = word_embd_dim
        self.embd_dim = embd_dim
        self.drop = nn.Dropout(drop_rate)
        self.num_layers = num_layers
        self.heads = heads
        self.hidden_size = hidden_size
        self.num_concepts = num_concepts

        # layers
        self.highway = Highway(h_size=embd_dim, h_out=embd_dim, num_layers=2)

        self.position = PositionalEncoding(d_model=embd_dim, dropout=drop_rate)
        self.segment = SegmentEncoding(d_model=embd_dim, dropout=drop_rate)

        attn = MHSEAttn(heads=heads, d_model=embd_dim, dropout=drop_rate)
        ff = PositionwiseFeedForward(d_model=embd_dim, d_ff=hidden_size, dropout=drop_rate)

        self.encoder = Encoder(layer = EncoderLayer(size=embd_dim, self_attn=c(attn),
                                                feed_forward=c(ff), dropout=drop_rate), N=num_layers)

        # projection layer
        self.classifier = Classifier(hidden_size=embd_dim, dropout=drop_rate)

        # initialize weights
        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            # nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Sequential):
            for p in module:
                self.initialize_weights(p)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def orthogonal_weights(self, module):
        """ Initialize the weights to be orthogonal.
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Sequential):
            for p in module:
                self.orthogonal_weights(p)

    def forward(self, qa, segment_ids, mask=None, concept=None):
        '''
         input : batch, seq_len
                qxlen [qx, lenx]
                qX: question one or two
                question: BxTxD
                memory:   BxTxD
                w1, w2: [B,T,D]
                qlength: [B, 1]
                qa_concept: [B, T1, T2, num_concepts]
                aq_concept: [B, T2, T1, num_concepts]
        '''
        c = copy.deepcopy

        qa = self.highway(qa)

        # input encoding
        qa = self.position(qa) # add positional encoding
        qa = self.segment(qa, segment_ids)

        # encoder
        qa = self.encoder(qa, mask, concept, 50.0) # encode with self-attention [B, T, D]

        # pooling
        qa_pool = qa[:,0,:].squeeze(1) # use CLS token as pooled vector [B, D]

        score = self.classifier(qa_pool) # [B, 3]

        return score
