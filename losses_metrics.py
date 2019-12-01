import numpy as np

from scipy.stats import spearmanr

from common import N_TARGETS
from utils.torch import to_numpy


def spearmanr_np(preds, targets):
    score = 0
    for i in range(N_TARGETS):
        score_i = spearmanr(preds[:, i], targets[:, i]).correlation
        score += np.nan_to_num(score_i / N_TARGETS)
    return score


def spearmanr_torch(preds, targets):
    return spearmanr_np(to_numpy(preds), to_numpy(targets))
