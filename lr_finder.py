import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim.lr_scheduler import LambdaLR

from utils.helpers import update_ewma_lst
from utils.torch import to_device


def expspace(s, e, n=100):
    return np.exp(np.linspace(np.log(s), np.log(e), n))


class LRFinder():
    EWMA_FACTOR = 0.9
    def __init__(self, start_lr:float=1e-6, end_lr:float=10, n_iter:int=100, 
                 divergence_factor:float=10, path:str='tmp/', 
                 device:torch.device=torch.device('cuda'), grad_accum=1):
        self.start_lr, self.end_lr = start_lr, end_lr
        self.n_iter, self.divergence_factor = n_iter, divergence_factor
        self.tmp_file = f'{path}tmp_pre_lr_finder.pth'
        self.device = device
        self.losses, self.lrs, self.smooth_losses = None, None, None
        self.grad_accum = grad_accum
        os.makedirs(path, exist_ok=True)
    
    def find_lr(self, model:Module, opt:torch.optim.Optimizer, dl:DataLoader, 
                loss_fn, plot:bool=True, skip_start:int=0, skip_end:int=5):
        torch.save({
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict()
        }, self.tmp_file)
        
        self.set_scheduler(opt)
        
        self.losses, self.smooth_losses = [], []
        data_iter, best_loss = iter(dl), 1e6
        for i in tqdm(range(self.n_iter)):
            inputs, targets = next(data_iter)
            inputs, targets = to_device(inputs, self.device), targets.to(self.device)
            
            preds = model(*inputs)
            loss = loss_fn(preds, targets)

            loss.backward()
            
            if (i == 0) or (loss < best_loss): best_loss = loss
            if (loss / self.divergence_factor) > best_loss: break
            
            if i % self.grad_accum == self.grad_accum - 1:
                opt.step()
                opt.zero_grad()
            self.sched.step()
            
            self.losses.append(loss.item())
            update_ewma_lst(self.smooth_losses, loss.item(), self.EWMA_FACTOR)
            
        self.lrs = self.lrs[:len(self.losses)]
        
        checkpoint = torch.load(self.tmp_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['opt_state_dict'])

        if plot: self.plot_losses(skip_start, skip_end)
    
    def set_scheduler(self, opt:torch.optim.Optimizer):
        lr_factors = expspace(1, self.end_lr/self.start_lr, n=self.n_iter+1)
        lr_lambda = lambda i: lr_factors[i]
        self.lrs =  [lr_lambda(i) * self.start_lr for i in range(self.n_iter)]
        init_lrs = [g['lr'] for g in opt.param_groups]
        rescale_factor = self.start_lr / init_lrs[0]
        for g in opt.param_groups: g['lr'] *= rescale_factor
        self.sched = LambdaLR(opt, lr_lambda)
        
    def plot_losses(self, skip_start:int=0, skip_end:int=5, smooth:bool=True):
        if (self.losses is not None) and (self.lrs is not None):
            losses = self.smooth_losses if smooth else self.losses
            s, e = skip_start, skip_end
            plt.figure()
            plt.title('LR-Finder')
            plt.ylabel('losses')
            plt.xlabel('base lr')
            plt.semilogx(self.lrs[s:][:-e], losses[s:][:-e])
        else:
            print('First run find_lr.')  
