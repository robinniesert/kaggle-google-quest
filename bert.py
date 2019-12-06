import torch
import torch.nn as nn

from transformers import DistilBertModel

from net import GELU
from common import *


class Head(nn.Module):
    def __init__(self, n_h=512, n_feats=75, n_emb=512, n_bert=768):
        super().__init__()
        n_x = n_feats + 3 * n_emb + 2 * n_bert
        self.lin = nn.Sequential(
            nn.Linear(n_x, n_h),
            GELU(),
            nn.Dropout(0.2),
        )
        self.lin_q = nn.Sequential(
            nn.Linear(n_feats + 2 * n_emb + n_bert, n_h),
            GELU(),
            nn.Dropout(0.2),
        )
        self.lin_a = nn.Sequential(
            nn.Linear(n_feats + n_emb + n_bert, n_h),
            GELU(),
            nn.Dropout(0.2)
        )
        self.head_q = nn.Linear(2 * n_h, N_Q_TARGETS)
        self.head_a = nn.Linear(2 * n_h, N_A_TARGETS)
        
    def forward(self, x):
        x_q = self.lin_q(torch.cat([x[0], x[1], x[2], x[4]], dim=1))
        x_a = self.lin_a(torch.cat([x[0], x[3], x[5]], dim=1))
        x = self.lin(torch.cat(x, dim=1))
        x_q = self.head_q(torch.cat([x, x_q], dim=1))
        x_a = self.head_a(torch.cat([x, x_a], dim=1))
        return torch.cat([x_q, x_a], dim=1)

    
class HeadNet(Head):
    def forward(self, x_feats, x_q_emb, x_t_emb, x_a_emb, x_q_bert, x_a_bert):
        return super().forward((x_feats, x_q_emb, x_t_emb, x_a_emb, x_q_bert, x_a_bert))


class CustomBert(nn.Module):
    def __init__(self, n_h):
        super().__init__()
        self.q_bert = DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=False, output_attentions=False)
        self.a_bert = DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=False, output_attentions=False)
        self.head = HeadNet(n_h)
    
    def forward(self, x_feats, x_q_emb, x_t_emb, x_a_emb, q_ids, a_ids, 
                q_att_mask, a_att_mask):
        x_q_bert = self.q_bert(q_ids, attention_mask=q_att_mask)[0][:, 0, :]
        x_a_bert = self.a_bert(a_ids, attention_mask=a_att_mask)[0][:, 0, :]
        return self.head(x_feats, x_q_emb, x_t_emb, x_a_emb, x_q_bert, x_a_bert)