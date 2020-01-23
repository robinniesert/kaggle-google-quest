
import torch
import torch.nn as nn
import transformers


class TransformerRanker(nn.Module):
    def __init__(self, dim=128, bs=8, n_heads=8, n_layers=2):
        super().__init__()
        self.bs = bs
        self.proj = nn.Linear(1, dim)
        self.config = transformers.configuration_distilbert.DistilBertConfig(
            dim=dim, hidden_dim=int(4*dim), n_heads=n_heads, n_layers=n_layers, 
            dropout=0.0, attention_dropout=0.0)
        self.transformer = transformers.modeling_distilbert.Transformer(self.config)
        self.head = nn.Linear(dim, 1)
        
    def forward(self, x):
        mask = torch.ones(x.size(), dtype=torch.uint8, device=x.device)
        x = self.proj(x.unsqueeze(-1))
        x = self.transformer(x, mask, [None] * self.config.num_hidden_layers)[0]
        x = self.head(x).view(x.size(0), -1)
        return (torch.sigmoid(x) * 1.2) - .1 + (1 / 16)


class GRURanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(1, 32, 6, batch_first=True, bidirectional=True)
        self.sig = nn.Sigmoid()

    def forward(self, input_):
        input_ = input_.unsqueeze(-1)
        x, hn = self.rnn(input_)
        out = x.sum(dim=2)
        return self.sig(out).squeeze()
    
    
class ExactRanker(nn.Module):
    EPS = 1e-10
    def __init__(self):
        super().__init__()

    def comp(self, inpu):
        in_mat1 = torch.triu(inpu.repeat(inpu.size(0), 1), diagonal=1)
        in_mat2 = torch.triu(inpu.repeat(inpu.size(0), 1).t(), diagonal=1)

        comp_first = (in_mat1 - in_mat2)
        comp_second = (in_mat2 - in_mat1)

        std1 = torch.std(comp_first).item() + self.EPS
        std2 = torch.std(comp_second).item() + self.EPS

        comp_first = torch.sigmoid(comp_first * (6.8 / std1))
        comp_second = torch.sigmoid(comp_second * (6.8 / std2))

        comp_first = torch.triu(comp_first, diagonal=1)
        comp_second = torch.triu(comp_second, diagonal=1)

        return (torch.sum(comp_first, 1) + torch.sum(comp_second, 0) + 1) / inpu.size(0)

    def forward(self, input_):
        out = [self.comp(input_[d]) for d in range(input_.size(0))]
        out = torch.stack(out)
        return out.view(input_.size(0), -1)


class LSTMRanker(nn.Module):
    def __init__(self, seq_len=8):
        super().__init__()
        self.lstm = nn.LSTM(1, 512, 2, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv1d(seq_len, seq_len, 1024)

    def forward(self, input_):
        input_ = input_.reshape(input_.size(0), -1, 1)
        out, _ = self.lstm(input_)
        out = self.conv1(out)
        return out.view(input_.size(0), -1)