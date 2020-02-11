import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import spearmanr, rankdata

from utils.torch import to_numpy
from common import N_TARGETS, TARGETS


def scale(x, d):
    if d: return (x//(1/d))/d
    else: return x


def ahmet_round(preds, ds, indices):
    new_preds = preds.copy()
    for idx, d in zip(indices, ds):
        new_preds[:,idx] = scale(preds[:,idx], d)
    return new_preds


ds = [2, 4, 8, 16, 32, 64]


def optimize_rounding_params(oofs, y, verbose=True, ix=None):
    ix = ix if ix is not None else np.arange(oofs.shape[0])
    opt_ds = []
    opt_indices = []
    for idx in range(N_TARGETS):
        scores = [np.nan_to_num(spearmanr(scale(oofs[ix,idx], d), y[ix,idx])[0]) for d in ds]
        opt_d = ds[np.argmax(scores)]
        if ((np.max(scores) - spearmanr(oofs[ix,idx], y[ix,idx])[0]) > 0.002):
            if verbose: print(idx, opt_d, np.max(scores))
            opt_ds.append(opt_d)
            opt_indices.append(idx)
    return opt_ds, opt_indices


def optimized_ahmet_round(oofs, y, verbose=True, ix=None):
    return ahmet_round(oofs, *optimize_rounding_params(oofs, y, verbose, ix))


hard_targets = ['question_not_really_a_question', 'question_type_spelling']
def spearmanr_np(preds, targets, ix=None, ignore_hard_targets=False, optimized_rounding=False):
    ix = ix if ix is not None else np.arange(preds.shape[0])
    n_targets = N_TARGETS - ignore_hard_targets * len(hard_targets)
    if optimized_rounding:
        preds = optimized_ahmet_round(preds, targets, verbose=False, ix=ix)
    score = 0
    for i, t in enumerate(TARGETS):
        if ignore_hard_targets and t in hard_targets: 
            continue
        score_i = spearmanr(preds[ix, i], targets[ix, i]).correlation
        score += np.nan_to_num(score_i / n_targets) 
    return score


def get_cvs(oofs, y, ix):
    spearmanrs = [
        spearmanr_np(oofs, y),
        spearmanr_np(oofs, y, ix=ix),
        spearmanr_np(oofs, y, ignore_hard_targets=True),
        spearmanr_np(oofs, y, ix=ix, ignore_hard_targets=True),
        spearmanr_np(oofs, y, optimized_rounding=True),
        spearmanr_np(oofs, y, ix=ix, optimized_rounding=True),
        spearmanr_np(oofs, y, ignore_hard_targets=True, optimized_rounding=True),
        spearmanr_np(oofs, y, ix=ix, ignore_hard_targets=True, optimized_rounding=True)
    ]
    index = [
        'CV',
        'CV unique rows',
        'CV ignoring hard targets',
        'CV unique rows ignoring hard targets',
        'CV w rounding',
        'CV unique rows w rounding',
        'CV ignoring hard targets w rounding',
        'CV unique rows ignoring hard targets w rounding'
    ]
    return pd.DataFrame(spearmanrs, index=index, columns=['SpearmanR'])


def spearmanr_torch(preds, targets):
    return spearmanr_np(to_numpy(torch.sigmoid(preds)), to_numpy(targets))


def binary_cross_entropy(input, target, eps=1e-7):
    n = input.size(0)
    input = torch.clamp(input, min=eps, max=1-eps).view(n, -1)
    target = target.view(n, -1)
    loss = - (target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()
    return loss


class SCELoss(nn.Module):
    def __init__(self, alpha=1.0, A=-4, logits=True):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.exp_A = np.exp(A)
        self.logits = logits

    def forward(self, input, target):
        input = torch.sigmoid(input) if self.logits else input
        ce = binary_cross_entropy(input, target)

        target = torch.clamp(target.float(), min=self.exp_A, max=1 - self.exp_A)
        rce = binary_cross_entropy(target, input)

        loss = (self.alpha * ce + rce) / (self.alpha + 1)
        return loss