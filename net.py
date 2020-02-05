import torch
import torch.nn as nn

from common import N_TARGETS, N_Q_TARGETS, N_A_TARGETS

    
class GELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)



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



class CustomBert2(nn.Module):
    def __init__(self, n_h, n_feats):
        super().__init__()
        self.q_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.a_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.head = Head2(n_h, n_feats)
    
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
        self.head = Head2(n_h, n_cats, n_bert)

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
    att_mask = x_ids > 0
    x_bert = bert(x_ids, attention_mask=att_mask)[0]
    att_mask = att_mask.unsqueeze(-1)
    return (x_bert * att_mask).sum(dim=1) / att_mask.sum(dim=1)


def apply_bert_batch(ids, n_seqs, n_bert, bert, seq_add):
    bs = n_seqs.size(0)
    x_bert = torch.zeros((bs, n_bert), dtype=torch.float, device=ids.device)
    n_seqs_exp = torch.cat([n.repeat(n) for n in n_seqs])
    
    one_idx = n_seqs == 1
    one_idx_exp = n_seqs_exp == 1

    idxs = torch.arange(bs)
    idxs_exp = torch.cat([torch.full((n,), i)  for i, n in enumerate(n_seqs)])
    
    if torch.any(one_idx): x_bert[one_idx] = apply_bert(ids[one_idx_exp], bert)
        
    for idx in idxs[~one_idx]:
        for i, id in enumerate(ids[idxs_exp==idx]):
            if i == 0:
                x_bert[idx] = apply_bert(id.view(1, -1), bert).squeeze() / n_seqs[idx].float()
            else:
                x_bert[idx] += apply_bert(id.view(1, -1), bert).squeeze() / n_seqs[idx].float()
                #torch.cat((x_bert[idx].view(1,-1), apply_bert(id.view(1, -1), bert))).max(dim=0).values # / n_seqs[idx].float()
        
    return x_bert


class AddSeq(nn.Module):
    def __init__(self, n_h):
        super().__init__()
        self.lin = nn.Linear(n_h, n_h)
        self.ln = nn.LayerNorm(n_h)
        # self.apply(self._init_weights)

    def forward(self, x1, x2):
        return x1 + x2

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class CustomBert7(nn.Module):
    def __init__(self, n_h, n_feats, n_bert=768):
        super().__init__()
        self.n_bert = n_bert
        self.q_add = AddSeq(n_bert)
        self.a_add = AddSeq(n_bert)
        self.q_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.a_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.head = HeadNet2(n_h, n_feats, n_bert=n_bert)
    
    def forward(self, x_feats, q_ids, a_ids, n_q_seqs, n_a_seqs):
        x_q_bert = apply_bert_batch(q_ids, n_q_seqs, self.n_bert, self.q_bert, self.q_add)
        x_a_bert = apply_bert_batch(a_ids, n_a_seqs, self.n_bert, self.a_bert, self.a_add)
        return self.head(x_feats, x_q_bert, x_a_bert)



class AttFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(in_features=dim, out_features=hidden_dim)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=1)
        self.activation = GELU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.ln(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x


class Attention2(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True):
        super().__init__()
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        self.ffn = AttFFN(feature_dim, 4*feature_dim)
        if bias: self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        eij = self.ffn(x).view(-1, self.step_dim)
        if self.bias: eij = eij + self.b
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        if mask is not None: a = a * mask
        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * torch.unsqueeze(a, -1)
        return weighted_input.sum(dim=1)

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
    
    
class NeuralNet5(nn.Module):
    def __init__(self,
                 hidden_size: int = 128,
                 max_len: int = 500,
                 max_len_title: int = 30,
                 n_cat: int = 3,
                 cat_emb: int = 6,
                 n_host: int = 55,
                 host_emb: int = 28,
                 embedding_matrix=None):
        super(NeuralNet5, self).__init__()

        n_word = embedding_matrix.shape[1]
        self.embedding = nn.Embedding(*embedding_matrix.shape)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding = nn.Embedding(*embedding_matrix.shape)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = SpatialDropout(0.3)
        
        self.category_embedding = nn.Embedding(n_cat, int(cat_emb))
        self.host_embedding = nn.Embedding(n_host, int(host_emb))

        self.linear_q_add = nn.Linear(n_word, hidden_size)
        self.linear_q_add1 = nn.Linear(hidden_size, hidden_size//4)

        self.lstm_q = nn.LSTM(n_word, hidden_size, bidirectional=True, batch_first=True)
        self.gru_q = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_a = nn.LSTM(n_word, hidden_size, bidirectional=True, batch_first=True)
        self.gru_a = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_t = nn.LSTM(n_word, hidden_size, bidirectional=True, batch_first=True)
        self.gru_t = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention_q = Attention(hidden_size * 2, max_len)
        self.gru_attention_q = Attention(hidden_size * 2, max_len)

        self.lstm_attention_a = Attention(hidden_size * 2, max_len)
        self.gru_attention_a = Attention(hidden_size * 2, max_len)

        self.lstm_attention_t = Attention(hidden_size * 2, max_len_title)
        self.gru_attention_t = Attention(hidden_size * 2, max_len_title)

        self.linear_q = nn.Linear(hidden_size * 8, hidden_size//2)
        self.relu_q = GELU()

        self.linear_a = nn.Linear(hidden_size * 8, hidden_size//2)
        self.relu_a = GELU()

        self.linear_t = nn.Linear(hidden_size * 8, hidden_size//2)
        self.relu_t = GELU()
        
        self.linear_q_emb = nn.Linear(512, hidden_size//2)
        self.relu_q_emb = GELU()

        self.linear_a_emb = nn.Linear(512, hidden_size//2)
        self.relu_a_emb = GELU()

        self.linear_t_emb = nn.Linear(512, hidden_size//2)
        self.relu_t_emb = GELU()

        self.linear1 = nn.Sequential(
            nn.Linear(2 * hidden_size + int(cat_emb) + int(host_emb) + 6, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2), nn.ReLU(inplace=True), nn.Dropout(0.5)
        )
        self.linear_q_out = nn.Linear(hidden_size//2, N_Q_TARGETS)

        self.bilinear = nn.Bilinear(hidden_size//2, hidden_size//2, hidden_size//2)
        self.bilinear_emb = nn.Bilinear(hidden_size//2, hidden_size//2, hidden_size//2)
        self.linear2 = nn.Sequential(nn.Linear(3 * hidden_size + 6, hidden_size//2),
                                     nn.BatchNorm1d(hidden_size//2),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.5))

        self.linear_aq_out = nn.Linear(hidden_size//2,  N_A_TARGETS)
        
        self.bilinear_add = nn.Bilinear(hidden_size//4, hidden_size//4, N_TARGETS)

    def forward(self, question, answer, title, category, host, use_emb_q, use_emb_a, use_emb_t, dist_feature):
        h_embedding_q = self.embedding(question)
        h_embedding_q = self.embedding_dropout(h_embedding_q)

        h_lstm_q, _ = self.lstm_q(h_embedding_q)
        h_gru_q, _ = self.gru_q(h_lstm_q)

        h_lstm_atten_q = self.lstm_attention_q(h_lstm_q)
        h_gru_atten_q = self.gru_attention_q(h_gru_q)

        avg_pool_q = torch.mean(h_gru_q, 1)
        max_pool_q, _ = torch.max(h_gru_q, 1)

        h_embedding_a = self.embedding(answer)
        h_embedding_a = self.embedding_dropout(h_embedding_a)

        h_lstm_a, _ = self.lstm_a(h_embedding_a)
        h_gru_a, _ = self.gru_a(h_lstm_a)

        h_lstm_atten_a = self.lstm_attention_a(h_lstm_a)
        h_gru_atten_a = self.gru_attention_a(h_gru_a)

        avg_pool_a = torch.mean(h_gru_a, 1)
        max_pool_a, _ = torch.max(h_gru_a, 1)

        h_embedding_t = self.embedding(title)
        h_embedding_t = self.embedding_dropout(h_embedding_t)

        h_lstm_t, _ = self.lstm_t(h_embedding_t)
        h_gru_t, _ = self.gru_t(h_lstm_t)

        h_lstm_atten_t = self.lstm_attention_t(h_lstm_t)
        h_gru_atten_t = self.gru_attention_t(h_gru_t)

        avg_pool_t = torch.mean(h_gru_t, 1)
        max_pool_t, _ = torch.max(h_gru_t, 1)

        category = self.category_embedding(category)
        host = self.host_embedding(host)
        
        add = torch.cat((h_embedding_q, h_embedding_a, h_embedding_t), 1)
        add = self.linear_q_add(torch.mean(add, 1))
        add = self.linear_q_add1(add)

        q = torch.cat((h_lstm_atten_q, h_gru_atten_q, avg_pool_q, max_pool_q), 1)
        a = torch.cat((h_lstm_atten_a, h_gru_atten_a, avg_pool_a, max_pool_a), 1)
        t = torch.cat((h_lstm_atten_t, h_gru_atten_t, avg_pool_t, max_pool_t), 1)
        
        q = self.relu_q(self.linear_q(q))
        a = self.relu_a(self.linear_a(a))
        t = self.relu_t(self.linear_t(t))

        q_emb = self.relu_q_emb(self.linear_q_emb(use_emb_q))
        a_emb = self.relu_a_emb(self.linear_a_emb(use_emb_a))
        t_emb = self.relu_t_emb(self.linear_t_emb(use_emb_t))
        
        hidden_q = self.linear1(torch.cat((q, t, q_emb, t_emb, category, host, dist_feature), 1))
        q_result = self.linear_q_out(hidden_q)

        bil_sim = self.bilinear(q, a)
        bil_sim_emb = self.bilinear_emb(q_emb, a_emb)
        
        hidden_aq = self.linear2(torch.cat((q, a, q_emb, a_emb, bil_sim, bil_sim_emb, dist_feature), 1))
        aq_result = self.linear_aq_out(hidden_aq)

        out = torch.cat([q_result, aq_result], 1)
        out = self.bilinear_add(out, add)

        return out


def linear_bn_act_drop(n_in, n_out, act=GELU(), bn=False, drop=0.0):
    layers = [nn.Linear(n_in, n_out)]
    if bn: layers.append(nn.BatchNorm1d(n_out))
    layers.append(act)
    if drop > 1e-5: layers.append(nn.Dropout(drop))
    return nn.Sequential(*layers)
