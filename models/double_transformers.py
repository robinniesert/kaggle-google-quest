import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AlbertModel

from models.head import Head
from models.siamese_transformers import avg_pool_forward
from common import *


class DoubleTransformer(nn.Module):
    def __init__(self, AvgPooledModel, pretrained_model_name='albert-base-v2'):
        super().__init__()
        self.q_transformer = AvgPooledModel.from_pretrained(pretrained_model_name)
        self.a_transformer = AvgPooledModel.from_pretrained(pretrained_model_name)
        self.head = Head(n_h=256, n_feats=5, n_bert=768, dropout=0.2)
    
    def forward(self, x_feats, q_ids, a_ids, seg_q_ids=None, seg_a_ids=None):
        x_q = self.q_transformer(q_ids, seg_q_ids)
        x_a = self.a_transformer(a_ids, seg_a_ids)
        return self.head(x_feats, x_q, x_a)


class AvgPooledAlbert(AlbertModel):
    def forward(self, ids, seg_ids=None):
        return avg_pool_forward(self, AvgPooledAlbert, ids, seg_ids)
    
    
class DoubleAlbert(DoubleTransformer):
    def __init__(self):
        super().__init__(AvgPooledAlbert, 'albert-base-v2')