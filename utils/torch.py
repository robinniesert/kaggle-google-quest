import torch
import torch.nn as nn
from scipy.stats import rankdata

from common import *


def to_cpu(x):
    return x.contiguous().detach().cpu()


def to_numpy(x):
    return to_cpu(x).numpy()


def to_device(xs, device, no_wrap_for_singles=False):
    if isinstance(xs, tuple) or isinstance(xs, list):
        return [to_device(x, device, no_wrap_for_singles=True) for x in xs]
    else: 
        if no_wrap_for_singles: return xs.to(device)
        else: return [xs.to(device)]


def set_optimizer_mom(opt, mom):
    has_betas = 'betas' in opt.param_groups[0]
    has_mom = 'momentum' in opt.param_groups[0]
    if has_betas:
        for g in opt.param_groups:
            _, beta = g['betas']
            g['betas'] = (mom, beta)
    elif has_mom:
        for g in opt.param_groups:
            g['momentum'] = mom