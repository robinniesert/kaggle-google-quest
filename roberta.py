import math
import copy
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaModel

from bert import GELU, Head2
from common import *


class AvgPooledRoberta(RobertaModel):
    def forward(self, ids, seg_ids=None):
        att_mask = ids > 0
        x_bert = super().forward(ids, att_mask, token_type_ids=seg_ids)[0]
        att_mask = att_mask.unsqueeze(-1)
        return (x_bert * att_mask).sum(dim=1) / att_mask.sum(dim=1)
    
    def resize_type_embeddings(self, new_num_types):
        old_embeddings = self.embeddings.token_type_embeddings
        model_embeds = self._get_resized_embeddings(old_embeddings, new_num_types)
        self.embeddings.token_type_embeddings = model_embeds
        self.config.type_vocab_size = new_num_types
        self.type_vocab_size = new_num_types
    
    
class CustomRoberta(nn.Module):
    def __init__(self, n_h, n_feats, head_dropout=0.2):
        super().__init__()
        self.roberta = AvgPooledRoberta.from_pretrained('roberta-base')
        self.roberta.resize_type_embeddings(2)
        self.head = Head2(n_h, n_feats, n_bert=768, dropout=head_dropout)
    
    def forward(self, x_feats, q_ids, a_ids, seg_q_ids=None, seg_a_ids=None):
        x_q_bert = self.roberta(q_ids, seg_q_ids)
        x_a_bert = self.roberta(a_ids, seg_a_ids)
        return self.head(x_feats, x_q_bert, x_a_bert)


class CLSPooledRoberta(RobertaModel):
    def forward(self, ids, seg_ids=None):
        att_mask = ids > 0
        return super().forward(ids, att_mask, token_type_ids=seg_ids)[0][:,0,:]
    
    def resize_type_embeddings(self, new_num_types):
        old_embeddings = self.embeddings.token_type_embeddings
        model_embeds = self._get_resized_embeddings(old_embeddings, new_num_types)
        self.embeddings.token_type_embeddings = model_embeds
        self.config.type_vocab_size = new_num_types
        self.type_vocab_size = new_num_types
    
    
class CustomRoberta2(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.roberta = CLSPooledRoberta.from_pretrained('roberta-base')
        self.roberta.resize_type_embeddings(3)
        self.head = nn.Linear(768 + n_feats, N_TARGETS)
    
    def forward(self, x_feats, ids, seg_ids=None):
        x_bert = self.roberta(ids, seg_ids)
        return self.head(torch.cat([x_feats, x_bert], dim=1))