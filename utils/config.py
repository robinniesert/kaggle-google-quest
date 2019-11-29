import numpy as np
import pandas as pd

import importlib
import copy

import albumentations as albu
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cloud_data import *
from common import *


def get_cfg_attr(key, config, default=None):
    if key in config: return config[key]
    else: return default


def build_augmentation(tranform_config, type='train'): 
    return albu.load(tranform_config[type.upper()])


def adjust_final_conv(model, type_idx):
    final_conv = model.decoder.final_conv
    model.decoder.final_conv = torch.nn.Conv2d(
        model.decoder.final_conv.in_channels, 1, kernel_size=1, padding=0)
    model.decoder.final_conv.weight.data.copy_(final_conv.weight.data[type_idx])
    model.decoder.final_conv.bias.data.copy_(final_conv.bias.data[type_idx])


def adjust_final_linear(model, encoder):
    if encoder.find('resnet') != -1: 
        model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    if encoder == 'mobilenet_v2': 
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, N_CLASSES),
        )
    if encoder.find('densenet') != -1: 
        model.classifier = nn.Linear(model.classifier.in_features, N_CLASSES)
    else:
        raise ValueError(f'adjust_final_layer not defined for encoder: {encoder}')


def build_model(cfg, name=None, cloud_type=None):
    model_config = cfg.model_config
    model = object_from_config(model_config)
    if cfg.task == 'classification': adjust_final_linear(model, cfg.encoder)
    if 'MODEL_INIT_PATH' in model_config:
        if model_config['MODEL_INIT_PATH'] is not None:
            assert name is not None
            checkpoint_file = f"{model_config['MODEL_INIT_PATH']}{name}_best.pth"
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
    if cloud_type: adjust_final_conv(model, CLASSES.index(cloud_type))
    if cfg.use_gpu: model.cuda()
    return model


def get_test_ids(cfg):
    sub = pd.read_csv(f'{cfg.raw_data_dir}sample_submission.csv')
    return sub['Image_Label'].apply(
        lambda x: x.split('_')[0]).drop_duplicates().values


def build_loader(cfg, fold_id=None, mode='train', remove_bad_imgs=False, 
                 label_df=None):
    assert mode in ['train', 'val', 'test']
    
    if mode != 'test':
        img_ids = pd.read_csv(
            f'folds/{mode}_ids_5_fold_cv.csv', usecols=[0, fold_id], 
            index_col=0
        ).dropna().iloc[:,0].values
        if remove_bad_imgs: img_ids = np.setdiff1d(img_ids, BAD_IMGS)
    else:
        img_ids = get_test_ids(cfg)

    if cfg.pseudo_dir and (mode == 'train'):
        pseudo_ids = get_test_ids(cfg)
    else:
        pseudo_ids = None

    if mode == 'val': mode = 'valid'
    aug_mode = 'valid' if mode == 'test' else mode
    transforms = build_augmentation(cfg.transforms_config, type=aug_mode)

    if cfg.task == 'segmentation':
        dataset = CloudDataset(
            mode, img_ids, cfg.proc_data_dir, transforms, cfg.mean, cfg.std,
            pseudo_ids, pseudo_masks_folder=cfg.pseudo_dir)
    else:
        dataset = CloudClassDataset(
            mode, img_ids, label_df, cfg.proc_data_dir, transforms, cfg.mean, 
            cfg.std
        )
    shuffle = mode == 'train'
    loader = DataLoader(
        dataset, batch_size=cfg.bs, shuffle=shuffle, 
        num_workers=cfg.n_workers
    )

    return img_ids, loader


def build_model_specs(cfg, fold_id):
    if cfg.n_top_models:
        assert cfg.top_epochs is not None
        assert fold_id in cfg.top_epochs

        cfgs = []
        for epoch in cfg.top_epochs[fold_id]:
            cfg_epoch = copy.deepcopy(cfg)
            cfg_epoch.use_epoch = epoch
            cfgs.append(cfg_epoch)
        model_names = cfg.n_top_models * [f'{cfg.model_name}_fold_{fold_id}']
    else: 
        cfgs = [cfg]
        model_names = [f'{cfg.model_name}_fold_{fold_id}']

    return cfgs, model_names


def get_model_spec_str(cfg):
    model_spec = '_'.join(cfg.model_name.split('_')[1:])
    if cfg.weight_averaging: model_spec += '_swa'
    if cfg.n_top_models: model_spec += f'_top_{cfg.n_top_models}'
    if cfg.pseudo_dir: model_spec += '_pseudo'
    model_spec += '_{}_{}'.format(*cfg.proc_img_sz)
    return model_spec


def object_from_config(config, init=True):
    module = importlib.import_module(config['MODULE'])
    if 'CLASS' in config:
        obj = getattr(module, config['CLASS'])
        if init:
            kwargs = config['KWARGS'] or {}
            obj = obj(**kwargs)
    elif 'FN' in config: obj = getattr(module, config['FN'])
    else: raise AttributeError('Config should include a CLASS or FN definition')
    return obj


class BaseConfig:
    def __init__(self, config):
        self.raw_data_dir = config['RAW_DATA_DIR']
        self.proc_data_dir = config['PROC_DATA_DIR']
        self.log_dir = config['LOG_DIR']
        self.sub_dir = config['SUB_DIR']
        self.checkpoint_dir = config['CHECKPOINT_DIR']
        self.post_proc_param_dir = config['POST_PROC_PARAM_DIR']
        self.pseudo_dir = get_cfg_attr('PSEUDO_DIR', config)

        self.device = torch.device(config['DEVICE'])
        self.use_gpu = config['DEVICE'].find('cuda') != -1
        self.n_workers = config['N_WORKERS']
        
        self.bs = config['BATCH_SIZE']
        self.proc_img_sz = config['PROC_IMG_SZ']
        self.proc_data_dir += 'images_{}_{}/'.format(*self.proc_img_sz)
        self.checkpoint_dir += '{}_{}/'.format(*self.proc_img_sz)

        self.task = get_cfg_attr('TASK', config, default='segmentation')

        self.ensemble_weight = get_cfg_attr('ENSEMBLE_WEIGHT', config, default=1)

        self.model_config = config['MODEL']
        if self.task == 'segmentation':
            self.seg_model_type = self.model_config['CLASS'].lower()
            self.encoder = self.model_config['KWARGS']['encoder_name']
            self.attn = get_cfg_attr('attention_type', self.model_config['KWARGS'])
            default_model_kwargs = {
                'encoder_weights': 'imagenet', 
                'classes': N_CLASSES, 
                'activation': None
            }
            for k, v in default_model_kwargs.items(): 
                self.model_config['KWARGS'][k] = v

            self.model_name = f'segmentation_{self.seg_model_type}_{self.encoder}'
            if self.attn is not None: self.model_name += f'_attn_{self.attn}'
                
            self.class_model = config['CLASS_MODEL']
        else:
            self.encoder = self.model_config['CLASS']
            self.model_name = f'classification_{self.encoder}'
        
        if self.encoder.find('inception') != -1: 
            self.mean, self.std = INCEPTION_MEAN, INCEPTION_STD
        if self.encoder.find('dpn') == -1: 
            self.mean, self.std = DPN_MEAN, DPN_STD
        else: 
            self.mean, self.std = MEAN, STD

        self.weight_averaging = self.model_config['SWA']

        self.n_top_models = config['N_TOP_MODELS']
        if self.n_top_models:
            self.top_epochs, self.use_epoch = {}, None
            if 'TOP_EPOCHS' in config: 
                if config['TOP_EPOCHS']: self.top_epochs = config['TOP_EPOCHS']
            
        self.ttas = config['TTA']

        self.transforms_config = config['TRANSFORMS']

        self.plot_preds = get_cfg_attr('PLOT_PREDICTIONS', config, False)