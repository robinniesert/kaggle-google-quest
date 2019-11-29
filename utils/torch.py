import torch.nn as nn


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