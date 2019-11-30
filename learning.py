import os
import copy
import matplotlib.pyplot as plot

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from utils.plotting import twin_plot
from utils.helpers import update_avg, update_ewma_lst

class Learner():
    EWMA_FACTOR = 0.9
    
    def __init__(self, model, optimizer, train_loader, valid_loader, loss_fn, 
                 device, n_epochs, model_name, checkpoint_dir, scheduler=None, 
                 metric_fns={}, monitor_metric='loss', minimize_score=True, 
                 logger=None, grad_accum=1, grad_clip=100.0, early_stopping=None, 
                 batch_step_scheduler=True, weight_averaging=False, 
                 eval_at_start=False, n_top_models=None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.n_epochs = n_epochs
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = scheduler
        self.metric_fns = metric_fns
        self.monitor_metric = monitor_metric
        self.minimize_score = minimize_score
        self.logger = logger
        self.grad_accum = grad_accum
        self.grad_clip = grad_clip
        self.early_stopping = early_stopping
        self.batch_step_scheduler = batch_step_scheduler
        self.weight_averaging = weight_averaging
        self.eval_at_start = eval_at_start
        self.n_top_models = n_top_models 
        if self.n_top_models: self.top_epochs, self.top_scores = [], []

        self.best_epoch, self.best_score = -1, 1e6 if minimize_score else -1e6
        self.lrs, self.scores = [], []
        self.train_losses, self.valid_losses = [], []
        self.smooth_train_losses = []
        self.train_metrics = {k: [] for k in self.metric_fns}
        self.valid_metrics = {k: [] for k in self.metric_fns}
        self.smooth_train_metrics  = {k: [] for k in self.metric_fns}

        if self.monitor_metric == 'loss':
            self.batch_update_score_as_loss = True
        else:
            monitor_update_type = self.metric_fns[self.monitor_metric][1]
            self.batch_update_score_as_loss = monitor_update_type != 'batch_end'
    
    @property
    def best_checkpoint_file(self): 
        return f'{self.checkpoint_dir}{self.model_name}_best.pth'

    @property
    def swa_checkpoint_file(self): 
        return f'{self.checkpoint_dir}{self.model_name}_swa.pth'

    def train(self):
        self.model.to(self.device)
        if self.weight_averaging: 
            self.swa_model = copy.deepcopy(self.model)
            self.swa_model.to(self.device)
        
        if self.eval_at_start:
            epoch = -1
            self.logger.info('epoch {}: \t Start validation...'.format(epoch))
            self.model.eval()
            val_score, val_loss, val_metrics = self.valid_epoch()
            self.logger.info(self._get_metric_string(
                epoch, val_loss, val_metrics, 'valid'))
            
            self.best_score, self.best_epoch = val_score, epoch
            self.save_model(self.best_checkpoint_file)
            self.logger.info(
                'best model: epoch {} - {:.5}'.format(epoch, val_score))

            if self.n_top_models:
                self._update_top_models(epoch, val_score)

        for epoch in range(self.n_epochs):
            self.logger.info('epoch {}: \t Start training...'.format(epoch))

            self.train_preds, self.train_targets = [], []
            self.valid_preds, self.valid_targets = [], []

            self.model.train()
            train_loss, train_metrics = self.train_epoch()
            self.logger.info(self._get_metric_string(
                epoch, train_loss, train_metrics))
            
            self.logger.info('epoch {}: \t Start validation...'.format(epoch))
            self.model.eval()
            val_score, val_loss, val_metrics = self.valid_epoch()
            self.logger.info(self._get_metric_string(
                epoch, val_loss, val_metrics, 'valid'))
            
            if ((self.minimize_score and (val_score < self.best_score)) or 
                ((not self.minimize_score) and (val_score > self.best_score))):
                self.best_score, self.best_epoch = val_score, epoch
                self.save_model(self.best_checkpoint_file)
                self.logger.info(
                    'best model: epoch {} - {:.5}'.format(epoch, val_score))

            if self.n_top_models: 
                self._update_top_models(epoch, val_score)

            if self.weight_averaging:
                self._update_average_model(epoch)
            
            self.lrs.append(self.optimizer.param_groups[0]['lr'])
            
            if not self.batch_step_scheduler: self._step_scheduler(val_score)
            
            if self.early_stopping is not None:
                if epoch - self.best_epoch > self.early_stopping:
                    self.logger.info('EARLY STOPPING')
                    self._on_training_end()
                    return
                
        self._on_training_end()
            
    def train_epoch(self):
        tqdm_loader = tqdm(self.train_loader)
        curr_loss_avg, curr_metric_avgs = 0, {k: 0 for k in self.metric_fns}

        for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
            inputs, targets = self.to_device(inputs), targets.to(self.device)
            preds, loss = self.train_batch(inputs, targets, batch_idx)
            
            self.train_preds.append(preds)
            self.train_targets.append(targets)

            curr_loss_avg, curr_metric_avgs = self._update_metrics(
                curr_loss_avg, loss, curr_metric_avgs, preds, targets, 
                batch_idx
            )
            
            base_lr = self.optimizer.param_groups[0]['lr']
            tqdm_loader.set_description('loss: {:.4} base_lr: {:.6}'.format(
                round(curr_loss_avg, 4), round(base_lr, 6)))

        for k, (metric_fn, update_type) in self.metric_fns:
            if update_type == 'epoch_end':
                metric_val = metric_fn(self.train_preds, self.train_targets).item()
                curr_metric_avgs[k] = metric_val
                self.train_metrics[k].append(metric_val)

        return curr_loss_avg, curr_metric_avgs
    
    def valid_epoch(self):
        tqdm_loader = tqdm(self.valid_loader)
        curr_loss_avg, curr_metric_avgs = 0, {k: 0 for k in self.metric_fns}
        
        for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
            with torch.no_grad():
                inputs, targets = self.to_device(inputs), targets.to(self.device)
                preds, loss = self.valid_batch(inputs, targets)

                self.valid_preds.append(preds)
                self.valid_targets.append(targets)

                curr_loss_avg, curr_metric_avgs = self._update_metrics(
                    curr_loss_avg, loss, curr_metric_avgs, preds, targets, 
                    batch_idx, train=False
                )
                if self.batch_update_score_as_loss: score = curr_loss_avg
                else: score = curr_metric_avgs[self.monitor_metric]
                
                tqdm_loader.set_description(
                    'score: {:.4}'.format(round(score, 4)))

        for k, (metric_fn, update_type) in self.metric_fns:
            if update_type == 'epoch_end':
                metric_val = metric_fn(self.valid_preds, self.valid_targets).item()
                curr_metric_avgs[k] = metric_val
                if self.monitor_metric==k: score = metric_val
        
        self.scores.append(score)
        self.valid_losses.append(curr_loss_avg)
        for k in self.metric_fns: 
            self.valid_metrics[k].append(curr_metric_avgs[k])

        return score, curr_loss_avg, curr_metric_avgs
    
    def train_batch(self, batch_inputs, batch_targets, batch_idx):
        preds, loss = self.get_loss_batch(batch_inputs, batch_targets)

        loss.backward()

        if batch_idx % self.grad_accum == self.grad_accum - 1:
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.batch_step_scheduler: self._step_scheduler()
        return preds, loss.item()
    
    def valid_batch(self, batch_inputs, batch_targets):
        preds, loss = self.get_loss_batch(batch_inputs, batch_targets)
        return preds, loss.item()
    
    def get_loss_batch(self, batch_inputs, batch_targets):
        preds = self.model(batch_inputs)
        loss = self.loss_fn(preds, batch_targets)
        return preds, loss

    def to_device(self, xs):
        if isinstance(xs, tuple) or isinstance(xs, list):
            return [x.to(self.device) for x in xs]
        else: return xs.to(self.device)
    
    def load_best_model(self, epoch=None):
        if epoch: checkpoint = torch.load(self.checkpoint_file(epoch))
        else: checkpoint = torch.load(self.best_checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def save_model(self, checkpoint_file):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_file)

    def checkpoint_file(self, epoch): 
        return f'{self.checkpoint_dir}{self.model_name}_epoch_{epoch}.pth'

    def _get_metric_string(self, epoch, loss, metrics, stage='train'):
        base_str = 'epoch {}/{} \t {} : loss {:.5}'.format(
            epoch, self.n_epochs, stage, loss)
        return base_str + ''.join(' - {} {:.5}'.format(k, v) 
                                  for k, v in metrics.items())

    def _update_top_models(self, epoch, score):
        self.save_model(self.checkpoint_file(epoch))
        self.top_scores.append((score, epoch))
        if len(self.top_scores) > self.n_top_models:
            if self.minimize_score:
                _, rm_epoch = self.top_scores.pop(
                    self.top_scores.index(max(self.top_scores)))
            else:
                _, rm_epoch = self.top_scores.pop(
                    self.top_scores.index(min(self.top_scores)))

            if rm_epoch != epoch:
                self.top_epochs = [e for _, e in self.top_scores]
                self.logger.info(
                    f'Updated top {self.n_top_models} models: '
                    f'removing epoch {rm_epoch} - new top epochs {self.top_epochs}'
                )

    def _update_average_model(self, epoch):
        pars = self.model.state_dict()
        pars_swa = self.swa_model.state_dict()
        with torch.no_grad():
            for k in pars.keys():
                pars_swa[k].data.copy_(
                    update_avg(pars_swa[k].data, pars[k].data, epoch))
    
    def _update_metrics(self, curr_loss_avg, loss, curr_metric_avgs, 
                        preds, targets, batch_idx, train=True):
        curr_loss_avg = update_avg(curr_loss_avg, loss, batch_idx)
        if train:
            self.train_losses.append(loss)
            update_ewma_lst(self.smooth_train_losses, loss, self.EWMA_FACTOR)
        
        for k, (metric_fn, update_type) in self.metric_fns:
            if update_type == 'batch_end':
                metric_val = metric_fn(preds, targets).item()
                curr_metric_avgs[k] = update_avg(
                    curr_metric_avgs[k], metric_val, batch_idx)
                if train:
                    self.train_metrics[k].append(metric_val)
                    update_ewma_lst(self.smooth_train_metrics[k], metric_val, 
                                    self.EWMA_FACTOR)
        
        return curr_loss_avg, curr_metric_avgs
    
    def _step_scheduler(self, score=None):
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau): 
                self.scheduler.step(score)
            else: self.scheduler.step()
            
    def _on_training_end(self):
        if self.weight_averaging:
            self.logger.info(
                'epoch {}: \t Start SWA validation...'.format(self.n_epochs))
            self._update_batch_norm()
            self.save_model(self.swa_checkpoint_file)
            self.model = copy.deepcopy(self.swa_model)
            self.model.to(self.device)
            _, val_loss, val_metrics = self.valid_epoch()
            self.logger.info(self._get_metric_string(
                self.n_epochs, val_loss, val_metrics, 'SWA'))
        if self.n_top_models:
            for epoch in range(-int(self.eval_at_start), self.n_epochs):
                if epoch not in self.top_epochs:
                    os.remove(self.checkpoint_file(epoch))
        self.logger.info('TRAINING END: Best score achieved on epoch '
                         f'{self.best_epoch} - {self.best_score:.5f}')
            
    def _update_batch_norm(self):
        # run one forward pass of train data to update batch norm running stats
        self.swa_model.train()
        for inputs, _ in tqdm(self.train_loader):
            inputs = self.to_device(inputs)
            self.swa_model(inputs)
        self.swa_model.eval()

    def plot_losses(self, smooth=True):
        train_vals = self.smooth_train_losses if smooth else self.train_losses
        twin_plot(train_vals, self.valid_losses, ax1_label='Step', 
                  ax2_label='Epoch', y_label='Loss', label1='train', 
                  label2='valid')
        
    def plot_lr(self):
        plt.figure()
        plt.title('LR')
        plt.xlabel('Epoch')
        plt.plot(self.lrs)
    
    def plot_metric(self, metric_name, smooth=True):
        k = metric_name
        ax1_label = 'Step' if self.metric_fns[k][1] == 'batch_end' else 'Epoch'
        if smooth: train_vals = self.smooth_train_metrics[k] 
        else: train_vals = self.train_metrics[k]
        twin_plot(train_vals, self.valid_metrics[k], ax1_label=ax1_label, 
                  ax2_label='Epoch', y_label=k, label1='train', label2='valid')
 