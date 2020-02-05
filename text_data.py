import numpy as np
import random
import torch
from torch.utils.data import Dataset

from common import N_TARGETS


class TextDataset3(Dataset):

    def __init__(self, x_features, question_ids, answer_ids, idxs, targets=None):
        self.question_ids = question_ids[idxs].astype(np.long)
        self.answer_ids = answer_ids[idxs].astype(np.long)
        self.x_features = x_features[idxs].astype(np.float32)
        if targets is not None: self.targets = targets[idxs].astype(np.float32)
        else: self.targets = np.zeros((self.x_features.shape[0], N_TARGETS), dtype=np.float32)

    def __getitem__(self, idx):
        q_ids = self.question_ids[idx]
        a_ids = self.answer_ids[idx]
        x_feats = self.x_features[idx]
        target = self.targets[idx]
        return (x_feats, q_ids, a_ids), target

    def __len__(self):
        return len(self.x_features)


class TextDataset4(Dataset):

    def __init__(self, x_features, ids, seg_ids, idxs, targets=None):
        self.ids = ids[idxs].astype(np.long)
        self.seg_ids = seg_ids[idxs].astype(np.long)
        self.x_features = x_features[idxs].astype(np.float32)
        if targets is not None: self.targets = targets[idxs].astype(np.float32)
        else: self.targets = np.zeros((self.x_features.shape[0], N_TARGETS), dtype=np.float32)

    def __getitem__(self, idx):
        ids = self.ids[idx]
        seg_ids = self.seg_ids[idx]
        x_feats = self.x_features[idx]
        target = self.targets[idx]
        return (x_feats, ids, seg_ids), target

    def __len__(self):
        return len(self.x_features)



class TextDataset5(Dataset):

    def __init__(self, x_features, question_ids, answer_ids, seg_question_ids, 
                 seg_answer_ids, idxs, targets=None):
        self.question_ids = question_ids[idxs].astype(np.long)
        self.answer_ids = answer_ids[idxs].astype(np.long)
        self.seg_question_ids = seg_question_ids[idxs].astype(np.long)
        self.seg_answer_ids = seg_answer_ids[idxs].astype(np.long)
        self.x_features = x_features[idxs].astype(np.float32)
        if targets is not None: self.targets = targets[idxs].astype(np.float32)
        else: self.targets = np.zeros((self.x_features.shape[0], N_TARGETS), dtype=np.float32)

    def __getitem__(self, idx):
        q_ids = self.question_ids[idx]
        a_ids = self.answer_ids[idx]
        seg_q_ids = self.seg_question_ids[idx]
        seg_a_ids = self.seg_answer_ids[idx]
        x_feats = self.x_features[idx]
        target = self.targets[idx]
        return (x_feats, q_ids, a_ids, seg_q_ids, seg_a_ids), target

    def __len__(self):
        return len(self.x_features)


class BertDataset(Dataset):

    def __init__(self, x_features, question_outputs, answer_outputs, idxs, 
                 targets=None):
        self.question_outputs = question_outputs.astype(np.float32)
        self.answer_outputs = answer_outputs.astype(np.float32)
        self.x_features = x_features[idxs].astype(np.float32)
        if targets is not None: self.targets = targets[idxs].astype(np.float32)
        else: self.targets = np.zeros((self.x_features.shape[0], N_TARGETS), dtype=np.float32)

    def __getitem__(self, idx):
        q_outputs = self.question_outputs[idx]
        a_outputs = self.answer_outputs[idx]
        x_feats = self.x_features[idx]
        target = self.targets[idx]
        return (x_feats, q_outputs, a_outputs), target

    def __len__(self):
        return len(self.x_features)
