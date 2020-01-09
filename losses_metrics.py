import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import spearmanr

from common import N_TARGETS
from utils.torch import to_numpy


def my_round(x, num, dec=2):
    return np.round(x / num, dec) * num


def round_preds(preds, thres=0.0, low_dec=1, low_num=1, high_dec=2, high_num=3):
    low_idx = preds < thres
    new_preds = np.zeros_like(preds)
    new_preds[low_idx] = my_round(preds[low_idx], low_num, low_dec)
    new_preds[~low_idx] = my_round(preds[~low_idx], high_num, high_dec)
    return new_preds


def spearmanr_np(preds, targets):
    score = 0
    for i in range(N_TARGETS):
        score_i = spearmanr(preds[:, i], targets[:, i]).correlation
        score += np.nan_to_num(score_i / N_TARGETS)
    return score


def spearmanr_torch(preds, targets):
    return spearmanr_np(to_numpy(torch.sigmoid(preds)), to_numpy(targets))



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, input, target):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                input, target, reduce=False)
        else: BCE_loss = F.binary_cross_entropy(input, target, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduce: return torch.mean(F_loss)
        else: return F_loss


class MyRankingLoss(nn.MSELoss):
    def forward(self, input, target):
        input = torch.sigmoid(input)
        n = input.size(0)
        n_pairs = n // 2
        n_tot_pairs = n_pairs + (n % 2)
        loss = 0
        for i in range(n_pairs):
            dp = input[2*i] - input[(2*i)+1]
            dy = target[2*i] - target[(2*i)+1]
            loss += super().forward(dp, dy) / n_tot_pairs
            
        if n_tot_pairs > n_pairs:
            dp = input[-2] - input[-1]
            dy = target[-2] - target[-1]
            loss += super().forward(dp, dy) / n_tot_pairs
        return loss
    

class MixedLoss(nn.Module):
    def __init__(self, pos_weight=N_TARGETS*[1.0]):
        super().__init__()
        pos_weight = torch.Tensor(pos_weight).cuda()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
        self.mrl = MyRankingLoss()

    def forward(self, input, target):
        loss = (1. * self.bce(input, target) + 1. * self.mrl(input, target))
        return loss.mean()