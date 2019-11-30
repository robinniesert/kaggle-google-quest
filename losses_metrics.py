import numpy as np

from scipy.stats import spearmanr

from common import N_TARGETS

def to_numpy(x):
    return x.contiguos().detach().cpu().numpy()


def spearmanr_torch(preds, targets):
    preds, targets = to_numpy(preds), to_numpy(targets)
    score = 0
    for i in range(N_TARGETS):
        score_i = spearmanr(preds[:, i], targets[:, i]).correlation
        score += np.nan_to_num(score_i / N_TARGETS)
    return score
