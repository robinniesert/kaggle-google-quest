import math
import copy
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

from transformers import DistilBertModel, BertModel
from transformers.modeling_distilbert import MultiHeadSelfAttention

from net import GELU, Attention, Attention2
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


class MyMultiHeadSelfAttention(nn.Module):
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

    def forward(self, query, key, value, mask, query_mask=None):
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

        if query_mask is None: query_mask = torch.ones_like(query)
        else: query_mask = query_mask.unsqueeze(-1)

        q = shape(self.q_lin(query) * query_mask)           # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))             # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))           # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)                     # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2,3))          # (bs, n_heads, q_length, k_length)
        mask = (mask==0).view(mask_reshp).expand_as(scores) # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float('inf'))            # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)   # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)        # (bs, n_heads, q_length, k_length)

        context = torch.matmul(weights, v)     # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)             # (bs, q_length, dim)
        context = self.out_lin(context)        # (bs, q_length, dim)

        return context


class MyFFN(nn.Module):
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

        self.self_attention = MyMultiHeadSelfAttention(dim, n_heads, dropout)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

        if other_attention:
            self.other_attention = MyMultiHeadSelfAttention(dim, n_heads, dropout)
            self.oa_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)
        else:
            self.other_attention = None
            self.oa_layer_norm = None

        self.ffn = MyFFN(dim, hidden_dim, dropout)
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
    def __init__(self, n_layers, dim, hidden_dim, dropout, n_heads):
        super().__init__()
        self.n_layers = n_layers
        layer = MyTransformerBlock(dim, hidden_dim, dropout, n_heads)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, x, attn_mask=None):
        for layer_module in self.layers:
            x = layer_module(x=x, attn_mask=attn_mask)
        return x


class Head2(nn.Module):
    def __init__(self, n_h=512, n_feats=74, n_bert=768):
        super().__init__()
        n_x = n_feats + 2 * n_bert
        self.lin = nn.Sequential(
            nn.Linear(n_x, n_h),
            GELU(),
            nn.Dropout(0.2),
        )
        self.lin_q = nn.Sequential(
            nn.Linear(n_feats + n_bert, n_h),
            GELU(),
            nn.Dropout(0.2),
        )
        self.lin_a = nn.Sequential(
            nn.Linear(n_feats + n_bert, n_h),
            GELU(),
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
        self.head = HeadNet2(n_h, n_feats, n_bert=768)
    
    def forward(self, x_feats, q_ids, a_ids):
        q_att_mask = q_ids > 0
        a_att_mask = a_ids > 0
        x_q_bert = self.q_bert(q_ids, attention_mask=q_att_mask)[0]
        x_a_bert = self.a_bert(a_ids, attention_mask=a_att_mask)[0]
        q_att_mask = q_att_mask.unsqueeze(-1)
        a_att_mask = a_att_mask.unsqueeze(-1)
        x_q_bert = (x_q_bert * q_att_mask).sum(dim=1) / q_att_mask.sum(dim=1)
        x_a_bert = (x_a_bert * a_att_mask).sum(dim=1) / a_att_mask.sum(dim=1)
        return self.head(x_feats, x_q_bert, x_a_bert)


class CustomBert3b(nn.Module):
    def __init__(self, n_h, n_feats):
        super().__init__()
        n_bert = 768
        self.q_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.a_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.transformer = MyTransformer(1, n_bert, 4*n_bert, dropout=0.1, n_heads=12)
        self.head = HeadNet2(n_h, n_feats, n_bert=n_bert)
    
    def forward(self, x_feats, q_ids, a_ids):
        q_att_mask = q_ids > 0
        a_att_mask = a_ids > 0
        x_q_bert = self.q_bert(q_ids, attention_mask=q_att_mask)[0]
        x_a_bert = self.a_bert(a_ids, attention_mask=a_att_mask)[0]
        x_bert = torch.cat([x_q_bert, x_a_bert], dim=1)
        att_mask = torch.cat([q_att_mask, a_att_mask], dim=1)
        x_bert = self.transformer(x_bert, att_mask)
        x_q_bert = x_bert[:, :512]
        x_a_bert = x_bert[:, 512:]
        q_att_mask = q_att_mask.unsqueeze(-1)
        a_att_mask = a_att_mask.unsqueeze(-1)
        x_q_bert = (x_q_bert * q_att_mask).sum(dim=1) / q_att_mask.sum(dim=1)
        x_a_bert = (x_a_bert * a_att_mask).sum(dim=1) / a_att_mask.sum(dim=1)
        return self.head(x_feats, x_q_bert, x_a_bert)
   

class CustomBert4(nn.Module):
    def __init__(self, n_h, n_cats, bert_type='large'):
        super().__init__()
        self.bert = BertModel.from_pretrained(f'bert-{bert_type}-uncased')
        n_bert = 1024 if bert_type=='large' else 768
        self.head = HeadNet2(n_h, n_cats, n_bert)

    def forward(self, x_cats, ids, seg_ids):
        att_mask = ids > 0
        x_bert = self.bert(ids, att_mask, seg_ids)[0]
        seg_ids_q = (seg_ids.unsqueeze(-1) <= 1) * att_mask.unsqueeze(-1)
        seg_ids_a = (seg_ids.unsqueeze(-1) == 2) * att_mask.unsqueeze(-1)
        x_q_bert = (x_bert * seg_ids_q).sum(dim=1) / seg_ids_q.sum(dim=1)
        x_a_bert = (x_bert * seg_ids_a).sum(dim=1) / seg_ids_a.sum(dim=1)
        return self.head(x_cats, x_q_bert, x_a_bert)


def self_attent(transformer_block, x, attn_mask):
    out = transformer_block.attention(query=x, key=x, value=x, mask=attn_mask)[0]
    x = transformer_block.sa_layer_norm(out + x)
    return x


def apply_ffn(transformer_block, x):
    out = transformer_block.ffn(x)                             # (bs, seq_length, dim)
    x = transformer_block.output_layer_norm(out + x)
    return x


class NeighborAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.attention = MyMultiHeadSelfAttention(dim, n_heads, dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, x_neighbor, mask, neighbor_mask):
        neighbor_mask = neighbor_mask.unsqueeze(-1)
        x_neighbor = (x_neighbor * neighbor_mask).sum(dim=1) / neighbor_mask.sum(dim=1)
        x_neighbor = x_neighbor.unsqueeze(1).expand_as(x)
        out = self.attention(query=x_neighbor, key=x, value=x, mask=mask)
        x = self.layer_norm(out + x) 
        return x


class ParallelTransformer(nn.Module):
    def __init__(self, transformer1, transformer2, neighbor_layers=[0, 1, 2, 3, 4, 5]):
        super().__init__()
        self.layer1 = transformer1.layer
        self.layer2 = transformer2.layer
        self.neighbor_layers = neighbor_layers
        dim, n_heads, dropout = self.layer1[0].dim, self.layer1[0].n_heads, 0.1
        neigbor_attention = NeighborAttention(dim, n_heads, dropout)
        self.neighbor_attentions1 = nn.ModuleList(
            [copy.deepcopy(neigbor_attention) for _ in range(len(neighbor_layers))])
        self.neighbor_attentions2 = nn.ModuleList(
            [copy.deepcopy(neigbor_attention) for _ in range(len(neighbor_layers))])

    def forward(self, x1, x2, attn_mask1, attn_mask2):

        for i, (block1, block2) in enumerate(zip(self.layer1, self.layer2)):

            out1 = self_attent(block1, x1, attn_mask1)
            out2 = self_attent(block2, x2, attn_mask2)
            if i in self.neighbor_layers:
                idx = self.neighbor_layers.index(i)
                x1 = self.neighbor_attentions1[idx](out1, out2, attn_mask1, attn_mask2)
                x2 = self.neighbor_attentions2[idx](out2, out1, attn_mask2, attn_mask1)
            x1 = apply_ffn(block1, x1)
            x2 = apply_ffn(block2, x2)

        return x1, x2


class ParallelDistillBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert2 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.transformer = ParallelTransformer(self.bert1.transformer, self.bert2.transformer, [0,1,2,3,4,5])

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
        q_att_mask = q_att_mask.unsqueeze(-1)
        a_att_mask = a_att_mask.unsqueeze(-1)
        x_q_bert = (x_q_bert * q_att_mask).sum(dim=1) / q_att_mask.sum(dim=1)
        x_a_bert = (x_a_bert * a_att_mask).sum(dim=1) / a_att_mask.sum(dim=1)
        return self.head(x_feats, x_q_bert, x_a_bert)


max_seq_length = 512
max_q_length = 271
max_a_length = 241

def concat_q_and_a(q_x, a_x, q_mask, a_mask):
    d = q_x.device
    seqs = []
    new_masks = []
    new_q_masks = []
    new_a_masks = []
    for q, a, q_m, a_m in zip(q_x, a_x, q_mask, a_mask):
        n_q, n_a = q_m.sum(), a_m.sum()
        cut_off = n_q + n_a - max_seq_length
        if cut_off > 0:
            if n_q > max_q_length: n_q = max(max_q_length, n_q - cut_off)
            if n_a > max_a_length: n_a = max(max_a_length, n_a - cut_off)

        seqs.append(torch.cat([q[q_m==1][:n_q], a[a_m==1][:n_a]], dim=0))
        new_masks.append(torch.ones(n_q + n_a, device=d))
        new_q_masks.append(torch.ones(n_q, device=d))
        new_a_masks.append(torch.cat([torch.zeros(n_q, device=d), torch.ones(n_a, device=d)]))

    packed_seq = rnn.pack_sequence(seqs, enforce_sorted=False)
    packed_masks = rnn.pack_sequence(new_masks, enforce_sorted=False)
    packed_q_masks = rnn.pack_sequence(new_q_masks, enforce_sorted=False)
    packed_a_masks = rnn.pack_sequence(new_a_masks, enforce_sorted=False)
    padded_seq = rnn.pad_packed_sequence(packed_seq, batch_first=True, total_length=max_seq_length)[0]
    padded_masks = rnn.pad_packed_sequence(packed_masks, batch_first=True, total_length=max_seq_length)[0]
    padded_q_masks = rnn.pad_packed_sequence(packed_q_masks, batch_first=True, total_length=max_seq_length)[0]
    padded_a_masks = rnn.pad_packed_sequence(packed_a_masks, batch_first=True, total_length=max_seq_length)[0]

    return padded_seq, padded_masks.long(), padded_q_masks.unsqueeze(-1), padded_a_masks.unsqueeze(-1)


class CustomBert6(nn.Module):
    def __init__(self, n_h, n_feats):
        super().__init__()
        q_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        a_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.q_emb = copy.deepcopy(q_bert.embeddings)
        self.a_emb = copy.deepcopy(a_bert.embeddings)

        self.q_transformer = copy.deepcopy(q_bert.transformer.layer[:3])
        self.a_transformer = copy.deepcopy(a_bert.transformer.layer[:3])
        self.transformer = copy.deepcopy(bert.transformer.layer[3:])

        del q_bert, a_bert, bert
        gc.collect()

        self.head = HeadNet2(n_h, n_feats)#HeadNet3(n_h, n_feats)
    
    def forward(self, x_feats, q_ids, a_ids):
        q_att_mask = q_ids > 0
        a_att_mask = a_ids > 0
        
        q_x = self.q_emb(q_ids) 
        a_x = self.a_emb(a_ids)

        for i, (q_block, a_block) in enumerate(zip(self.q_transformer, self.a_transformer)):
            q_x = self_attent(q_block, q_x, q_att_mask)
            a_x = self_attent(a_block, a_x, a_att_mask)
            q_x = apply_ffn(q_block, q_x)
            a_x = apply_ffn(a_block, a_x)

        x, att_mask, q_att_mask, a_att_mask = concat_q_and_a(q_x, a_x, q_att_mask, a_att_mask)

        for i, block in enumerate(self.transformer):
            x = self_attent(block, x, att_mask)
            x = apply_ffn(block, x)

        x_q = (x * q_att_mask).sum(dim=1) / q_att_mask.sum(dim=1)
        x_a = (x * a_att_mask).sum(dim=1) / a_att_mask.sum(dim=1)
        return self.head(x_feats, x_q, x_a)


def apply_bert(x_ids, bert):
    att_mask = x_ids
    x_bert = bert(x_ids, attention_mask=att_mask)[0]
    att_mask = att_mask.unsqueeze(-1)
    return (x_bert * att_mask).sum(dim=1) / att_mask.sum(dim=1)


def convert_listy(l):
    if isinstance(l, list) or isinstance(l, tuple):
        return l[0]


class CustomBert7(nn.Module):
    def __init__(self, n_h, n_feats, n_bert=768):
        super().__init__()
        self.n_bert = n_bert
        self.q_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.a_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.head = HeadNet2(n_h, n_feats, n_bert=n_bert)
    
    def forward(self, x_feats, q_ids, a_ids, n_q_seqs, n_a_seqs):
        bs = x_feats.size(0)
        q_ids = convert_listy(q_ids)
        a_ids = convert_listy(a_ids)
        x_q_bert = torch.zeros((bs, self.n_bert), dtype=torch.float, device=x_feats.device)
        x_a_bert = torch.zeros((bs, self.n_bert), dtype=torch.float, device=x_feats.device)
        n_q_seqs_exp = torch.cat([n.repeat(n) for n in n_q_seqs])
        n_a_seqs_exp = torch.cat([n.repeat(n) for n in n_a_seqs])
        
        one_q_idx = n_q_seqs == 1
        one_a_idx = n_a_seqs == 1
        one_q_idx_exp = n_q_seqs_exp == 1
        one_a_idx_exp = n_a_seqs_exp == 1

        q_idxs = torch.arange(bs)
        a_idxs = torch.arange(bs)
        q_idxs_exp = torch.cat([torch.full((n,), i)  for i, n in enumerate(n_q_seqs)])
        a_idxs_exp = torch.cat([torch.full((n,), i)  for i, n in enumerate(n_a_seqs)])
        
        x_q_bert[one_q_idx] = apply_bert(q_ids[one_q_idx_exp], self.q_bert)
        x_a_bert[one_a_idx] = apply_bert(a_ids[one_a_idx_exp], self.a_bert)
        
        for q_idx in q_idxs[~one_q_idx]:
            x_q_bert[q_idx] = apply_bert(q_ids[q_idxs_exp==q_idx], self.q_bert).mean(dim=0)
        for a_idx in a_idxs[~one_a_idx]:
            x_a_bert[a_idx] = apply_bert(a_ids[a_idxs_exp==a_idx], self.a_bert).mean(dim=0)

        return self.head(x_feats, x_q_bert, x_a_bert)