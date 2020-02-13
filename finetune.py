import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW

from common import TARGETS, N_TARGETS
from utils.helpers import init_logger, init_seed
from utils.torch import to_numpy, to_device, to_cpu
from datasets import TextDataset, TransformerOutputDataset
from tokenization import tokenize
from learning import Learner
from one_cycle import OneCycleLR
from create_features import get_ohe_categorical_features
from losses_metrics import spearmanr_torch, get_cvs
from inference import infer

from train import models, pretrained_models, get_optimizer_param_groups


def get_model_outputs(model, loader, checkpoint_file, device, model_type='siamese'):

    q_outputs, a_outputs = [], []

    currently_deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True

    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(loader)):
            inputs = to_device(inputs, device)
            if model_type == 'siamese':
                batch_q_outputs = model.transformer(inputs[1], inputs[3])
                batch_a_outputs = model.transformer(inputs[2], inputs[4])
            if model_type == 'double':
                batch_q_outputs = model.q_transformer(inputs[1], inputs[3])
                batch_a_outputs = model.a_transformer(inputs[2], inputs[4])

            q_outputs.append(to_cpu(batch_q_outputs))
            a_outputs.append(to_cpu(batch_a_outputs))
        
        q_outputs = torch.cat(q_outputs)
        a_outputs = torch.cat(a_outputs)

    torch.backends.cudnn.deterministic = currently_deterministic

    return to_numpy(q_outputs), to_numpy(a_outputs)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Perform second stage of training - finetuning.')
    parser.add_argument('-model_name', type=str, default='siamese_roberta')
    parser.add_argument('-checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('-log_dir', type=str, default='logs/')
    parser.add_argument('-data_dir', type=str, default='data/')
    return parser


if __name__=='__main__':

    parser = build_parser()
    args = parser.parse_args()

    model_name = args.model_name
    model_type = 'double' if model_name == 'double_albert' else 'siamese'
    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    main_logger = init_logger(log_dir, f'finetune_main_{model_name}.log')


    # Import data
    test  = pd.read_csv(f'{args.data_dir}test.csv')
    train = pd.read_csv(f'{args.data_dir}train.csv')

    # Min Max scale target after rank transformation
    for col in TARGETS:
        train[col] = train[col].rank(method="average")
    train[TARGETS] = MinMaxScaler().fit_transform(train[TARGETS])
    y = train[TARGETS].values

    # Get model inputs
    ids_train, seg_ids_train = tokenize(train, pretrained_model_str=pretrained_models[model_name])
    cat_features_train, _ = get_ohe_categorical_features(train, test, 'category')


    # Set training parameters
    device       = 'cuda'
    num_workers  = 10
    n_folds      = 10
    lr           = 1e-5
    n_epochs     = 10
    bs           = 2
    grad_accum   = 4
    weight_decay = 0.01
    loss_fn      = nn.BCEWithLogitsLoss()

    # Start training
    init_seed()
    folds = GroupKFold(n_splits=n_folds).split(
        X=train['question_body'], groups=train['question_body'])
    oofs = np.zeros((len(train), N_TARGETS))

    main_logger.info(f'Start finetuning model {model_name}...')

    for fold_id, (train_index, valid_index) in enumerate(folds):
        
        main_logger.info(f'Fold {fold_id + 1} started at {time.ctime()}')
        
        fold_logger = init_logger(log_dir, f'finetune_fold_{fold_id+1}_{model_name}.log')
        
        loader = DataLoader(
            TextDataset(cat_features_train, ids_train['question'], ids_train['answer'],
                        seg_ids_train['question'], seg_ids_train['answer'], np.arange(len(train)), y), 
            batch_size=bs, shuffle=False, num_workers=num_workers
        )
        
        model = models[model_name]()
        checkpoint_file = f'{checkpoint_dir}{model_name}_fold_{fold_id+1}_best.pth'

        # Get last hidden layer outputs from transformers
        fold_logger.info(f'Precompute transformer outputs for model {model_name}...')
        q_outputs, a_outputs = get_model_outputs(model, loader, checkpoint_file, device, model_type)

        train_loader = DataLoader(
            TransformerOutputDataset(cat_features_train, q_outputs, a_outputs, train_index, y), 
            batch_size=bs, shuffle=True, num_workers=num_workers
        )
        valid_loader = DataLoader(
            TransformerOutputDataset(cat_features_train, q_outputs, a_outputs, valid_index, y),
            batch_size=bs, shuffle=False, num_workers=num_workers, drop_last=False
        )

        # Train the head of the model
        optimizer = AdamW(get_optimizer_param_groups(model.head, lr, weight_decay))

        learner = Learner(
            model.head, 
            optimizer, 
            train_loader, 
            valid_loader, 
            loss_fn, 
            device, 
            n_epochs, 
            f'{model_name}_head_fold_{fold_id + 1}', 
            checkpoint_dir, 
            scheduler=None, 
            metric_spec={'spearmanr': spearmanr_torch}, 
            monitor_metric=True,
            minimize_score=False, 
            logger=fold_logger,
            grad_accum=grad_accum,
            batch_step_scheduler=False,
            eval_at_start=True
        )
        learner.train()
        
        oofs[valid_index] = infer(learner.model, valid_loader, learner.best_checkpoint_file, device)

        # Save tuned model in half precision (reduces memory making it easier to upload to Kaggle)
        head_checkpoint_file = f'{checkpoint_dir}{model_name}_head_fold_{fold_id+1}_best.pth'
        checkpoint = torch.load(head_checkpoint_file)
        model.head.load_state_dict(checkpoint['model_state_dict'])
        model.half()
        tuned_checkpoint_file = f'{checkpoint_dir}{model_name}_tuned_fold_{fold_id+1}_best.pth'
        torch.save({'model_state_dict': model.state_dict()}, tuned_checkpoint_file)

        
    main_logger.info(f'Finished tuning {model_name}')


    # Print CV scores
    ix = np.where(train.groupby("question_body")["host"].transform("count")==1)[0] # unique question index
    main_logger.info('CVs:')
    main_logger.info(get_cvs(oofs, y, ix))

    # Store OOFs
    os.makedirs('oofs/', exist_ok=True)
    pd.DataFrame(oofs, columns=TARGETS).to_csv(f'oofs/{model_name}_tuned_oofs.csv')
