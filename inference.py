import numpy as np
import pandas as pd

from tqdm import tqdm_notebook as tqdm

import torch

from common import N_TARGETS
from utils.torch import to_device


def infer_batch(inputs, model, device, to_numpy=True):
    inputs = to_device(inputs, device)
    predicted = model(*inputs)
    inputs = [x.cpu() for x in inputs]
    preds = torch.sigmoid(predicted)
    if to_numpy: preds = preds.cpu().detach().numpy().astype(np.float32)
    return preds


def infer(model, loader, checkpoint_file=None, device=torch.device('cuda')):
    n_obs = len(loader.dataset)
    batch_sz = loader.batch_size
    predictions = np.zeros((n_obs, N_TARGETS))

    currently_deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True

    if checkpoint_file is not None:
        print(f'Starting inference for model: {checkpoint_file}')
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(loader)):
            start_index = i * batch_sz
            end_index = min(start_index + batch_sz, n_obs)
            batch_preds = infer_batch(inputs, model, device)
            predictions[start_index:end_index, :] += batch_preds

    torch.backends.cudnn.deterministic = currently_deterministic

    return predictions