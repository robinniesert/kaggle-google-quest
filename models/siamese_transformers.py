import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, XLNetModel, RobertaModel

from models.head import Head
from common import *


def avg_pool_forward(model, ModelClass, ids, seg_ids=None):
    att_mask = (ids > 0).float()
    x_out = super(ModelClass, model).forward(ids, att_mask, token_type_ids=seg_ids)[0]
    att_mask = att_mask.unsqueeze(-1)
    return (x_out * att_mask).sum(dim=1) / att_mask.sum(dim=1)


class SiameseTransformer(nn.Module):
    def __init__(self, AvgPooledModel, pretrained_model_name='bert-base-uncased'):
        super().__init__()
        self.transformer = AvgPooledModel.from_pretrained(pretrained_model_name)
        self.head = Head(n_h=256, n_feats=5, n_bert=768, dropout=0.2)
    
    def forward(self, x_feats, q_ids, a_ids, seg_q_ids=None, seg_a_ids=None):
        x_q = self.transformer(q_ids, seg_q_ids)
        x_a = self.transformer(a_ids, seg_a_ids)
        return self.head(x_feats, x_q, x_a)


class AvgPooledBert(BertModel):
    def forward(self, ids, seg_ids=None):
        return avg_pool_forward(self, AvgPooledBert, ids, seg_ids)
    
    
class SiameseBert(SiameseTransformer):
    def __init__(self):
        super().__init__(AvgPooledBert, 'bert-base-uncased')


class AvgPooledXLNet(XLNetModel):
    def forward(self, ids, seg_ids=None):
        return avg_pool_forward(self, AvgPooledXLNet, ids, seg_ids)
    
    
class SiameseXLNet(SiameseTransformer):
    def __init__(self):
        super().__init__(AvgPooledXLNet, 'xlnet-base-cased')


class AvgPooledRoberta(RobertaModel):
    def forward(self, ids, seg_ids=None):
        return avg_pool_forward(self, AvgPooledRoberta, ids, seg_ids)

    def resize_type_embeddings(self, new_num_types):
        old_embeddings = self.embeddings.token_type_embeddings
        model_embeds = self._get_resized_embeddings(old_embeddings, new_num_types)
        self.embeddings.token_type_embeddings = model_embeds
        self.config.type_vocab_size = new_num_types
        self.type_vocab_size = new_num_types
    
    
class SiameseRoberta(SiameseTransformer):
    def __init__(self):
        super().__init__(AvgPooledRoberta, 'roberta-base')
        self.transformer.resize_type_embeddings(2)