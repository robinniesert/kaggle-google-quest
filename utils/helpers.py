import os
import sys

import yaml
import json

import logging
import random
from pathlib import Path

import numpy as np
import torch


def init_seed(seed=100):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

    
def init_logger(directory, log_file_name):
    formatter = logging.Formatter( '\n%(asctime)s\t%(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    log_path = Path(directory, log_file_name)
    if log_path.exists(): log_path.unlink()
    handler = logging.FileHandler(filename=log_path)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def update_avg(curr_avg, val, idx):
    return (curr_avg * idx + val) / (idx + 1)
    

def update_ewma(prev_vals, val, factor):
    return val * (1 - factor) + prev_vals[-1] * factor
  

def update_ewma_lst(prev_vals, val, factor):
    if len(prev_vals)==0: prev_vals.append(val)
    else: prev_vals.append(update_ewma(prev_vals, val, factor))