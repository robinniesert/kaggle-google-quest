import os
import sys

import yaml
import json

import logging
import random
from pathlib import Path

import numpy as np
import torch
import cv2
import albumentations as albu

from common import CLASSES


def load_yaml(file_name):
    with open(file_name, 'r') as stream:
        result = yaml.load(stream, Loader=yaml.SafeLoader)
    return result


def load_json(file_name):
    with open(file_name, 'r') as stream:
        result = json.load(stream)
    return result


def init_seed(seed=100):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    
def init_logger(directory, log_file_name):
    formatter = logging.Formatter(
        '\n%(asctime)s\t%(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    log_path = Path(directory, log_file_name)
    if log_path.exists(): log_path.unlink()
    handler = logging.FileHandler(filename=log_path)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def get_img(img_name, img_folder):
    img_path = os.path.join(img_folder, img_name)
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_mask(img_name, masks_folder, shape=(320, 480)):
    masks = np.zeros(shape + (4, ), dtype=np.float32)
    for class_idx, class_id in enumerate(CLASSES):
        mask = cv2.imread(f'{masks_folder}{class_id}{img_name}', 0)
        if mask is None: continue
        if mask.shape!=shape: mask = cv2.resize(mask, shape[::-1])
        masks[:, :, class_idx] = mask
    masks /= 255
    return masks


def make_pseudo_mask(img_name, masks_folder, shape=(320, 480)):
    masks = np.zeros(shape + (4, ), dtype=np.float32)
    for class_idx, class_id in enumerate(CLASSES):
        mask = np.load(f'{masks_folder}{img_name}_{class_id}.npy')
        if mask.shape!=shape: mask = cv2.resize(mask, shape[::-1])
        masks[:, :, class_idx] = mask.astype(np.float32)
    return masks


def update_ewma(prev_vals, val, factor):
    return val * (1 - factor) + prev_vals[-1] * factor
  

def update_ewma_lst(prev_vals, val, factor):
    if len(prev_vals)==0: prev_vals.append(val)
    else: prev_vals.append(update_ewma(prev_vals, val, factor))


def update_avg(curr_avg, val, idx):
    return (curr_avg * idx + val) / (idx + 1)


def mask2rle(img):
    """
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)