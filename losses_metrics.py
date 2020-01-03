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
    preds = round_preds(preds)
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


class MixedLoss(nn.Module):
    def __init__(self, weight_bce=1, weight_fl=0, alpha=1, gamma=2,
                 pos_weight=N_TARGETS*[1.0]):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_fl = weight_fl
        pos_weight = torch.Tensor(pos_weight).cuda()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
        self.focal = FocalLoss(alpha, gamma, logits=True, reduce=True)

    def forward(self, input, target):
        input = input.transpose(1, 2).transpose(2, 3).contiguous()
        target = target.transpose(1, 2).transpose(2, 3).contiguous()
        loss = (self.weight_bce * self.bce(input, target) 
                + self.weight_fl * self.focal(input, target))
        return loss.mean()