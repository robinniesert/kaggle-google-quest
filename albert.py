import math
import copy
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AlbertModel

from bert import GELU, Head2
from common import *


class AvgPooledAlbert(AlbertModel):
    def forward(self, ids, seg_ids=None):
        att_mask = ids > 0
        x_bert = super().forward(ids, att_mask, token_type_ids=seg_ids)[0]
        att_mask = att_mask.unsqueeze(-1)
        return (x_bert * att_mask).sum(dim=1) / att_mask.sum(dim=1)
    
    
class CustomAlbert(nn.Module):
    def __init__(self, n_h, n_feats, head_dropout=0.2):
        super().__init__()
        self.q_albert = AvgPooledAlbert.from_pretrained('albert-large-v2')
        self.a_albert = AvgPooledAlbert.from_pretrained('albert-large-v2')
        self.head = Head2(n_h, n_feats, n_bert=1024, dropout=head_dropout)
    
    def forward(self, x_feats, q_ids, a_ids, seg_q_ids=None, seg_a_ids=None):
        x_q_bert = self.q_albert(q_ids, seg_q_ids)
        x_a_bert = self.a_albert(a_ids, seg_a_ids)
        return self.head(x_feats, x_q_bert, x_a_bert)

