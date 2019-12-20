import math
import copy

import torch
import torch.nn as nn

from transformers import DistilBertModel, BertModel
from transformers.modeling_distilbert import MultiHeadSelfAttention

from net import GELU
from common import *


class Head(nn.Module):
    def __init__(self, n_h=512, n_feats=74, n_emb=512, n_bert=768):
        super().__init__()
        n_x = n_feats + 3 * n_emb + 2 * n_bert
        self.lin = nn.Sequential(
            nn.Linear(n_x, n_h),
            GELU(),
            # nn.LayerNorm(n_h),
            nn.Dropout(0.2),
        )
        self.lin_q = nn.Sequential(
            nn.Linear(n_feats + 2 * n_emb + n_bert, n_h),
            GELU(),
            # nn.LayerNorm(n_h),
            nn.Dropout(0.2),
        )
        self.lin_a = nn.Sequential(
            nn.Linear(n_feats + n_emb + n_bert, n_h),
            GELU(),
            # nn.LayerNorm(n_h),
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
        self.q_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.a_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.head = HeadNet(n_h)
    
    def forward(self, x_feats, x_q_emb, x_t_emb, x_a_emb, q_ids, a_ids, 
                q_att_mask, a_att_mask):
        x_q_bert = self.q_bert(q_ids, attention_mask=q_att_mask)[0][:, 0, :]
        x_a_bert = self.a_bert(a_ids, attention_mask=a_att_mask)[0][:, 0, :]
        return self.head(x_feats, x_q_emb, x_t_emb, x_a_emb, x_q_bert, x_a_bert)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, attention_dropout):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.dropout = nn.Dropout(p=attention_dropout)
        assert self.dim % self.n_heads == 0
        self.q_lin = nn.Linear(in_features=dim, out_features=dim)
        self.k_lin = nn.Linear(in_features=dim, out_features=dim)
        self.v_lin = nn.Linear(in_features=dim, out_features=dim)
        self.out_lin = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, query, key, value, mask, head_mask = None):
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        dim_per_head = self.dim // self.n_heads
        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))           # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))             # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))           # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)                     # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2,3))          # (bs, n_heads, q_length, k_length)
        mask = (mask==0).view(mask_reshp).expand_as(scores) # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float('inf'))            # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)   # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)        # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)     # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)             # (bs, q_length, dim)
        context = self.out_lin(context)        # (bs, q_length, dim)

        return context


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(in_features=dim, out_features=hidden_dim)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.activation = GELU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class MyTransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, n_heads, other_attention=False):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)

        assert dim % n_heads == 0

        self.self_attention = MultiHeadSelfAttention(dim, n_heads, dropout)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

        if other_attention:
            self.other_attention = MultiHeadSelfAttention(dim, n_heads, dropout)
            self.oa_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)
        else:
            self.other_attention = None
            self.oa_layer_norm = None

        self.ffn = FFN(dim, hidden_dim, dropout)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

    def forward(self, x, x_other=None, attn_mask=None):
        # Self-Attention
        output = self.self_attention(query=x, key=x, value=x, mask=attn_mask)
        x = self.sa_layer_norm(output + x)          # (bs, seq_length, dim)

        # Other-Attention
        if x_other is not None:
            assert self.other_attention is not None
            output = self.other_attention(query=x_other, key=x, value=x, mask=attn_mask)
            x = self.sa_layer_norm(output + x)          # (bs, seq_length, dim)

        # Feed Forward Network
        output = self.ffn(x)                             # (bs, seq_length, dim)
        x = self.output_layer_norm(output + x)  # (bs, seq_length, dim)

        return x


class MyTransformer(nn.Module):
    def __init__(self, n_layers, dim, hidden_dim, dropout, n_heads, other_attention=False):
        super().__init__()
        self.n_layers = n_layers
        layer = MyTransformerBlock(dim, hidden_dim, dropout, n_heads, other_attention)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, x, attn_mask=None):
        hidden_state = x
        for layer_module in self.layers:
            layer_outputs = layer_module(x=hidden_state, attn_mask=attn_mask)
            hidden_state = layer_outputs
        return hidden_state


class Head2(nn.Module):
    def __init__(self, n_h=512, n_feats=74, n_bert=768):
        super().__init__()
        n_x = n_feats + 2 * n_bert
        self.lin = nn.Sequential(
            nn.Linear(n_x, n_h),
            GELU(),
            # nn.LayerNorm(n_h),
            nn.Dropout(0.2),
        )
        self.lin_q = nn.Sequential(
            nn.Linear(n_feats + n_bert, n_h),
            GELU(),
            # nn.LayerNorm(n_h),
            nn.Dropout(0.2),
        )
        self.lin_a = nn.Sequential(
            nn.Linear(n_feats + n_bert, n_h),
            GELU(),
            # nn.LayerNorm(n_h),
            nn.Dropout(0.2)
        )
        self.head_q = nn.Linear(2 * n_h, N_Q_TARGETS)
        self.head_a = nn.Linear(2 * n_h, N_A_TARGETS)
        
    def forward(self, x):
        x_q = self.lin_q(torch.cat([x[0], x[1]], dim=1))
        x_a = self.lin_a(torch.cat([x[0], x[2]], dim=1))
        x = self.lin(torch.cat(x, dim=1))
        x_q = self.head_q(torch.cat([x, x_q], dim=1))
        x_a = self.head_a(torch.cat([x, x_a], dim=1))
        return torch.cat([x_q, x_a], dim=1)


class HeadNet2(Head2):
    def forward(self, x_feats, x_q_bert, x_a_bert):
        return super().forward((x_feats, x_q_bert, x_a_bert))


class CustomBert2(nn.Module):
    def __init__(self, n_h, n_feats):
        super().__init__()
        self.q_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.a_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.head = HeadNet2(n_h, n_feats)
    
    def forward(self, x_feats, q_ids, a_ids):
        q_att_mask = q_ids > 0
        a_att_mask = a_ids > 0
        x_q_bert = self.q_bert(q_ids, attention_mask=q_att_mask)[0][:, 0, :]
        x_a_bert = self.a_bert(a_ids, attention_mask=a_att_mask)[0][:, 0, :]
        return self.head(x_feats, x_q_bert, x_a_bert)


class HeadNet4(nn.Module):
    def __init__(self, n_h_bert, n_cats):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(n_h_bert+n_cats, N_TARGETS)
        )
        self.q_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(n_h_bert+n_cats, N_Q_TARGETS)
        )
        self.a_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(n_h_bert+n_cats, N_A_TARGETS)
        )

    def forward(self, x_cats, x_bert, x_q_bert, x_a_bert):
        y = self.head(torch.cat([x_bert, x_cats], dim=1))
        y_q = self.q_head(torch.cat([x_q_bert, x_cats], dim=1))
        y_a = self.a_head(torch.cat([x_a_bert, x_cats], dim=1))
        return (y + torch.cat([y_q, y_a], dim=1)) / 2


class Head3(nn.Module):
    def __init__(self, n_h=512, n_feats=74, n_bert=768):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(n_feats + n_bert, n_h),
            GELU(),
            # nn.LayerNorm(n_h),
            nn.Dropout(0.2),
        )
        self.lin_q = nn.Sequential(
            nn.Linear(n_feats + n_bert, n_h),
            GELU(),
            # nn.LayerNorm(n_h),
            nn.Dropout(0.2),
        )
        self.lin_a = nn.Sequential(
            nn.Linear(n_feats + n_bert, n_h),
            GELU(),
            # nn.LayerNorm(n_h),
            nn.Dropout(0.2)
        )
        self.head_q = nn.Linear(2 * n_h, N_Q_TARGETS)
        self.head_a = nn.Linear(2 * n_h, N_A_TARGETS)
        
    def forward(self, x):
        x_q = self.lin_q(torch.cat([x[0], x[1]], dim=1))
        x_a = self.lin_a(torch.cat([x[0], x[2]], dim=1))
        x = self.lin(torch.cat([x[0], x[3]], dim=1))
        x_q = self.head_q(torch.cat([x, x_q], dim=1))
        x_a = self.head_a(torch.cat([x, x_a], dim=1))
        return torch.cat([x_q, x_a], dim=1)


class HeadNet3(nn.Module):
    def __init__(self, n_h=256, n_feats=74, n_h_bert=768):
        super().__init__()
        self.transformer = MyTransformer(
            1, n_h_bert, 4 * n_h_bert, dropout=0.1, n_heads=12)
        self.head = Head3(n_h, n_feats, n_h_bert)
        
    def forward(self, x_feats, x_q_bert, x_a_bert, att_mask):
        x = self.transformer(torch.cat([x_q_bert, x_a_bert], dim=1), att_mask)
        return self.head((x_feats, x_q_bert.mean(dim=1), x_a_bert.mean(dim=1), x.mean(dim=1)))


class CustomBert3(nn.Module):
    def __init__(self, n_h, n_feats):
        super().__init__()
        self.q_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.a_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.head = HeadNet2(n_h, n_feats)#HeadNet3(n_h, n_feats)
    
    def forward(self, x_feats, q_ids, a_ids):
        q_att_mask = q_ids > 0
        a_att_mask = a_ids > 0
        x_q_bert = self.q_bert(q_ids, attention_mask=q_att_mask)[0]
        x_a_bert = self.a_bert(a_ids, attention_mask=a_att_mask)[0]
        return self.head(x_feats, x_q_bert.mean(dim=1), x_a_bert.mean(dim=1))
        # att_mask = torch.cat([q_att_mask, a_att_mask], dim=1)
        # return self.head(x_feats, x_q_bert, x_a_bert, att_mask)
   

class CustomBert4(nn.Module):
    n_h_bert = 768
    def __init__(self, n_h, n_cats):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # self.head = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.n_h_bert+n_cats, N_TARGETS)
        # )
        self.head = HeadNet2(n_h, n_cats)
        # self.head = HeadNet4(self.n_h_bert, n_cats)

    def forward(self, x_cats, ids, seg_ids):
        att_mask = ids > 0
        x_bert = self.bert(ids, att_mask, seg_ids)[0]
        # x_bert = x_bert.mean(dim=1)
        # return self.head(torch.cat([x_bert, x_cats], dim=1))
        x_bert_q = (x_bert * (seg_ids.unsqueeze(-1) == 0)).mean(dim=1)
        x_bert_a = (x_bert * seg_ids.unsqueeze(-1)).mean(dim=1)
        return self.head(x_cats, x_bert_q, x_bert_a)
        # x_bert = x_bert.mean(dim=1)
        # return self.head(x_cats, x_bert, x_bert_q, x_bert_a)


def self_attent(transformer_block, x, attn_mask):
    out = transformer_block.attention(query=x, key=x, value=x, mask=attn_mask)
    x = transformer_block.sa_layer_norm(out + x)
    return x


def apply_ffn(transformer_block, x):
    out = transformer_block.ffn(x)                             # (bs, seq_length, dim)
    x = transformer_block.output_layer_norm(out + x)
    return x


class NeighborAttention(nn.Module):
    def __init__(self, dim, dropout, n_heads):
        super().__init__()
        self.attention = MultiHeadSelfAttention(dim, n_heads, dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

    def forward(self, x, x_neighbor, attn_mask):
        out = self.attention(query=x_neighbor, key=x, value=x, mask=attn_mask)
        x = self.layer_norm(out + x) 
        return x


class ParallelTransformer(nn.Module):
    def __init__(self, transformer1, transformer2):
        super().__init__()
        self.layer1 = transformer1.layer
        self.layer2 = transformer2.layer
        dim, dropout, n_heads = self.layer1[0].dim, 0.1, self.layer1[0].n_heads
        neigbor_attention = NeighborAttention(dim, n_heads, dropout)
        self.neighbor_attentions1 = nn.ModuleList(
            [copy.deepcopy(neigbor_attention) for _ in range(len(self.layer1))])
        self.neighbor_attentions2 = nn.ModuleList(
            [copy.deepcopy(neigbor_attention) for _ in range(len(self.layer2))])

    def forward(self, x1, x2, attn_mask1, attn_mask2):

        for block1, block2, neighbor_attn1, neighbor_attn2 in zip(
            self.layer1, self.layer2, self.neighbor_attentions1, self.neighbor_attentions2):

            out1 = self_attent(self.block1, x1, attn_mask1)
            out2 = self_attent(self.block2, x2, attn_mask2)
            x1 = neighbor_attn1(out1, out2, attn_mask1)
            x2 = neighbor_attn2(out2, out1, attn_mask2)
            x1 = apply_ffn(self.block1, x1)
            x2 = apply_ffn(self.block2, x2)

        return x1, x2


class ParallelDistillBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert2 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tansformer = ParallelTransformer(self.bert1.transformer, self.bert2.transformer)

    def forward(self, inp_ids1, inp_ids2, attn_mask1, attn_mask2):
        x1 = self.bert1.embeddings(inp_ids1) 
        x2 = self.bert2.embeddings(inp_ids2)
        x1, x2 = self.transformer(x1, x2, attn_mask1, attn_mask2)
        return x1, x2


class CustomBert5(nn.Module):
    def __init__(self, n_h, n_feats):
        super().__init__()
        self.parallel_bert = ParallelDistillBert()
        self.head = HeadNet2(n_h, n_feats) # HeadNet3(n_h, n_feats)
    
    def forward(self, x_feats, q_ids, a_ids):
        q_att_mask = q_ids > 0
        a_att_mask = a_ids > 0
        x_q_bert, x_a_bert = self.parallel_bert(q_ids, a_ids, q_att_mask, a_att_mask)
        return self.head(x_feats, x_q_bert.mean(dim=1), x_a_bert.mean(dim=1))