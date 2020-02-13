import math
import numpy as np
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import LambdaLR

from utils.torch import set_optimizer_mom


def cosine_annealing(it, n_iter, start_val, end_val):
    cos_inner = math.pi * (it % n_iter) / n_iter
    return ((start_val - end_val) * (math.cos(cos_inner) + 1) / 2) + end_val


def cosine_annealing_range(n_iter, start_val, end_val):
    return [cosine_annealing(i, n_iter, start_val, end_val) 
            for i in range(n_iter)]


class OneCycleLR(LambdaLR):
    def __init__(self, optimizer, lr_div_factor=25, warmup_frac=0.3, 
                 mom_range=(0.95, 0.85), n_epochs=10, n_batches=None, 
                 start_epoch=0):
        n_batches = 1 if n_batches is None else n_batches
        self.n_epochs, self.n_iter = n_epochs, (n_epochs * n_batches) + 1
        self.start_it = -1 if start_epoch==0 else start_epoch * n_batches
        self._build_schedules(lr_div_factor, mom_range, warmup_frac)
        super().__init__(optimizer, self.lr_lambda, last_epoch=self.start_it)
        
    def _build_schedules(self, lr_div_factor, mom_range, warmup_frac):
        n_warmup = int(self.n_iter * warmup_frac)
        n_decay = self.n_iter - n_warmup
        
        self.lrs = cosine_annealing_range(n_warmup, 1/lr_div_factor, 1)
        self.lrs += cosine_annealing_range(n_decay, 1, 1/lr_div_factor)
        self.lr_lambda = lambda i: self.lrs[i]
        
        self.moms = cosine_annealing_range(n_warmup, *mom_range)
        self.moms += cosine_annealing_range(n_decay, *mom_range[::-1])
        self.mom_lambda = lambda i: self.moms[i]
        
    def get_mom(self):
        return self.mom_lambda(self.last_epoch)

    def step(self, epoch=None):
        super().step(epoch)
        set_optimizer_mom(self.optimizer, self.get_mom())
        
    def plot_schedules(self):
        x = np.linspace(0, self.n_epochs, self.n_iter)
        _, ax = plt.subplots(1, 2, figsize=(15, 4))

        ax[0].set_title('LR Schedule')
        ax[0].set_ylabel('lr')
        ax[0].set_xlabel('epoch')
        ax[0].plot(x, self.lrs)

        ax[1].set_title('Momentum Schedule')
        ax[1].set_ylabel('momentum')
        ax[1].set_xlabel('epoch')
        ax[1].plot(x, self.moms)
