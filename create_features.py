import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_ohe_categorical_features(train, test, feature='category'):
    unique_vals = list(set(train[feature].unique().tolist() 
                           + test[feature].unique().tolist()))
    feat_dict = {i + 1: e for i, e in enumerate(unique_vals)}
    feat_dict_reverse = {v: k for k, v in feat_dict.items()}

    train_feat = train[feature].apply(lambda x: feat_dict_reverse[x]).values.reshape(-1, 1)
    test_feat = test[feature].apply(lambda x: feat_dict_reverse[x]).values.reshape(-1, 1)

    ohe = OneHotEncoder()
    ohe.fit(train_feat)
    train_feat = ohe.transform(train_feat).toarray()
    test_feat = ohe.transform(test_feat).toarray()

    return train_feat, test_feat
