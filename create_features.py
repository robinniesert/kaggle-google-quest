import numpy as np


def get_categorical_features(train, test, feature):
    unique_vals = list(set(train[feature].unique().tolist() 
                           + test[feature].unique().tolist()))
    feat_dict = {i + 1: e for i, e in enumerate(unique_vals)}
    feat_dict_reverse = {v: k for k, v in feat_dict.items()}

    train_feat = train[feature].apply(lambda x: feat_dict_reverse[x]).values
    test_feat = test[feature].apply(lambda x: feat_dict_reverse[x]).values

    return train_feat, test_feat, feat_dict, feat_dict_reverse
