import math
import copy

import torch
import torch.nn as nn

from transformers import DistilBertModel

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
        """
        Parameters
        ----------
        query: torch.tensor(bs, q_length, dim)
        key: torch.tensor(bs, k_length, dim)
        value: torch.tensor(bs, k_length, dim)
        mask: torch.tensor(bs, k_length)
        Outputs
        -------
        context: torch.tensor(bs, q_length, dim)
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

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
    """Down samples sequence length"""
    def __init__(self, dim, hidden_dim, dropout, n_heads, down_sample=32):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.down_sample = down_sample

        assert dim % n_heads == 0

        self.attention = MultiHeadSelfAttention(dim, n_heads, dropout)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

        self.ffn = FFN(dim, hidden_dim, dropout)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)
        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(query=x[:, ::self.down_sample], key=x, value=x, mask=attn_mask, head_mask=head_mask)
        sa_output = self.sa_layer_norm(sa_output + x)          # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)                             # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        return ffn_output


class MyTransformer(nn.Module):
    def __init__(self, n_layers, dim, hidden_dim, dropout, n_heads, down_sample=32):
        super().__init__()
        self.n_layers = n_layers

        layer = MyTransformerBlock(dim, hidden_dim, dropout, n_heads, down_sample)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, x, attn_mask=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.
        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        """

        hidden_state = x
        for layer_module in self.layers:
            layer_outputs = layer_module(x=hidden_state, attn_mask=attn_mask)
            hidden_state = layer_outputs

        outputs = hidden_state
        return outputs  # last-layer hidden state


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
    def __init__(self, n_h=256, n_feats=74, n_bert=768):
        super().__init__()
        self.transformer = MyTransformer(2, n_bert, 4 * n_bert, dropout=0.1, n_heads=12, down_sample=32)
        self.head = Head3(n_h, n_feats, n_bert)
        
    def forward(self, x_feats, x_q_bert, x_a_bert):
        x = self.transformer(torch.cat([x_q_bert, x_a_bert], dim=1))
        return self.head((x_feats, x_q_bert[:, 0, :], x_a_bert[:, 0, :], x[:, 0, :]))


class CustomBert3(nn.Module):
    def __init__(self, n_h, n_feats):
        super().__init__()
        self.q_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.a_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.head = HeadNet3(n_h, n_feats)
    
    def forward(self, x_feats, q_ids, a_ids):
        q_att_mask = q_ids > 0
        a_att_mask = a_ids > 0
        x_q_bert = self.q_bert(q_ids, attention_mask=q_att_mask)[0]
        x_a_bert = self.a_bert(a_ids, attention_mask=a_att_mask)[0]
        return self.head(x_feats, x_q_bert, x_a_bert)
   

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