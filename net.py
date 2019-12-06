import torch
import torch.nn as nn

from common import N_TARGETS


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

    
class GELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


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
