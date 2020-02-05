import math
import copy
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

from common import *


class GELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
    

def lin_layer(n_in, n_out, dropout):
    return nn.Sequential(nn.Linear(n_in, n_out), GELU(), nn.Dropout(dropout))
    

class Head2(nn.Module):
    def __init__(self, n_h=512, n_feats=74, n_bert=768, dropout=0.2):
        super().__init__()
        n_x = n_feats + 2 * n_bert
        self.lin = lin_layer(n_in=n_x, n_out=n_h, dropout=dropout)
        self.lin_q = lin_layer(n_in=n_feats + n_bert, n_out=n_h, dropout=dropout)
        self.lin_a = lin_layer(n_in=n_feats + n_bert, n_out=n_h, dropout=dropout)
        self.head_q = nn.Linear(2 * n_h, N_Q_TARGETS)
        self.head_a = nn.Linear(2 * n_h, N_A_TARGETS)
        
    def forward(self, x_feats, x_q_bert, x_a_bert):
        x_q = self.lin_q(torch.cat([x_feats, x_q_bert], dim=1))
        x_a = self.lin_a(torch.cat([x_feats, x_a_bert], dim=1))
        x = self.lin(torch.cat([x_feats, x_q_bert, x_a_bert], dim=1))
        x_q = self.head_q(torch.cat([x, x_q], dim=1))
        x_a = self.head_a(torch.cat([x, x_a], dim=1))
        return torch.cat([x_q, x_a], dim=1)


class AvgPooledBert(BertModel):
    def forward(self, ids, seg_ids=None):
        att_mask = ids > 0
        x_bert = super().forward(ids, att_mask, token_type_ids=seg_ids)[0]
        att_mask = att_mask.unsqueeze(-1)
        return (x_bert * att_mask).sum(dim=1) / att_mask.sum(dim=1)
    
    
class CustomBert3(nn.Module):
    def __init__(self, n_h, n_feats, head_dropout=0.2):
        super().__init__()
        self.bert = AvgPooledBert.from_pretrained('bert-base-uncased')
        self.head = Head2(n_h, n_feats, n_bert=768, dropout=head_dropout)
    
    def forward(self, x_feats, q_ids, a_ids, seg_q_ids=None, seg_a_ids=None):
        x_q_bert = self.bert(q_ids, seg_q_ids)
        x_a_bert = self.bert(a_ids, seg_a_ids)
        return self.head(x_feats, x_q_bert, x_a_bert)

