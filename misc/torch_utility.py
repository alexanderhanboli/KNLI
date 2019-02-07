from __future__ import  print_function, division
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class DictToObj:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_state(m):
    '''
      This code returns model states
    '''
    if m is None:
        return None
    state = {}
    for k, v in m.state_dict().items():
        state[k] = v.clone()
    return state

def load_model_states(path):
    '''
     Load previously learned model
    '''
    checkpoint = torch.load(path, map_location='cpu')
    m_states = checkpoint['model_states']
    m_params = checkpoint['args']

    return m_states, DictToObj(**m_params)

def to_contiguous(tensor):

    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class ReinforceCriterion(nn.Module):

    def __init__(self):
        super(ReinforceCriterion, self).__init__()

    def forward(self, actions, logprobs, reward):
        '''
            Actions  is B*C and one hot matrix
            logprobs is B*C
            where B is batch size and c is number of classes
        '''
        B = logprobs.size()[0]
        logprobs = to_contiguous(logprobs).view(-1)
        reward = to_contiguous(reward).view(-1)
        actions = to_contiguous(actions).view(-1).float()
        output = -1.0 * actions * logprobs * reward
        output = torch.sum(output) / B

        return output

class SmoothF1Criterion(nn.Module):

    def __init__(self):
        super(SmoothF1Criterion, self).__init__()

    def forward(self, logprobs, tp_masks, fp_masks, fn_masks, gt_masks):
        '''
            Actions  is B*C and one hot matrix
            logprobs is B*C
            where B is batch size and c is number of classes
        '''
        B = logprobs.size()[0]
        logprobs = to_contiguous(logprobs).view(-1)
        tp_masks = to_contiguous(tp_masks).view(-1).float()
        fp_masks = to_contiguous(fp_masks).view(-1).float()
        fn_masks = to_contiguous(fn_masks).view(-1).float()
        gt_masks = to_contiguous(gt_masks).view(-1).float()

        if torch.sum(tp_masks).data[0] > 0:
            pr =  torch.sum(tp_masks * logprobs) / (torch.sum(tp_masks * logprobs) + torch.sum(fp_masks * logprobs))
            re =  torch.sum(tp_masks * logprobs) / (torch.sum(tp_masks * logprobs) + torch.sum(fn_masks * logprobs))

            if math.isclose(pr.data[0], 0) or math.isclose(re.data[0], 0):
                f = torch.sum(gt_masks * logprobs)

            else:
                f  =  2.0 * pr * re / (pr + re)

        else:
            f = torch.sum(gt_masks * logprobs)

        output = -1.0 * f / B

        return output

def masked_softamx(x, mask, dim = 1):
    '''
        This function does masked-softmax
        inputs:
            x must be [b, seq_len, 1]
            mask must be [b, seq_length]
        returns:
            [b, seq_len, 1]

    '''
    x = x.squeeze(-1) # [b, seq_len, 1] ==> [b, seq_len]
    x = mask * x      # [b, seq_len]
    x = x / (x.sum(dim=dim, keepdim=True) + 1e-13)
    x = x.unsqueeze(-1) # [b, seq_len] ==> [b, seq_len, 1]

    return x

class PenalizedConfidenceEntropy(nn.Module):

    def __init__(self, beta = 0.1):
        super(PenalizedConfidenceEntropy, self).__init__()
        self.beta = beta

    def forward(self, preds):
        '''
            beta: controls the strength of the confidence penalty
            predictions is B*C
            where B is batch size and c is number of classes
        '''
        B = preds.size()[0]
        probs = F.softmax(preds, dim = -1 )
        logprobs = F.log_softmax(preds, dim = -1 )

        logprobs = to_contiguous(logprobs).view(-1)
        probs    = to_contiguous(probs).view(-1)


        output = -1.0 * self.beta * logprobs * probs
        output = torch.sum(output) / B

        return output
