import numpy as np


def get_dist_features(embedding_train, embedding_test):
    
    # define a square dist lambda function were (x1 - y1) ^ 2 + (x2 - y2) ^ 2 + (x3 - y3) ^ 2 + ... + (xn - yn) ^ 2
    # with this we get one vector of dimension 6079
    l2_dist = lambda x, y: np.power(x - y, 2).sum(axis = 1)
    
    # define a cosine dist lambda function were (x1 * y1) ^ 2 + (x2 * y2) + (x3 * y3) + ... + (xn * yn)
    cos_dist = lambda x, y: (x * y).sum(axis = 1)
    
    # transpose it because we have 6 vector of dimension 6079, need 6079 x 6
    dist_features_train = np.array([
        l2_dist(embedding_train['question_title_embedding'], 
                embedding_train['answer_embedding']),
        l2_dist(embedding_train['question_body_embedding'], 
                embedding_train['answer_embedding']),
        l2_dist(embedding_train['question_body_embedding'], 
                embedding_train['question_title_embedding']),
        cos_dist(embedding_train['question_title_embedding'], 
                 embedding_train['answer_embedding']),
        cos_dist(embedding_train['question_body_embedding'], 
                 embedding_train['answer_embedding']),
        cos_dist(embedding_train['question_body_embedding'], 
                 embedding_train['question_title_embedding'])
    ]).T
    
    # transpose it because we have 6 vector of dimension 6079, need 6079 x 6
    dist_features_test = np.array([
        l2_dist(embedding_test['question_title_embedding'], 
                embedding_test['answer_embedding']),
        l2_dist(embedding_test['question_body_embedding'], 
                embedding_test['answer_embedding']),
        l2_dist(embedding_test['question_body_embedding'], 
                embedding_test['question_title_embedding']),
        cos_dist(embedding_test['question_title_embedding'], 
                 embedding_test['answer_embedding']),
        cos_dist(embedding_test['question_body_embedding'], 
                 embedding_test['answer_embedding']),
        cos_dist(embedding_test['question_body_embedding'], 
                 embedding_test['question_title_embedding'])
    ]).T
    
    return dist_features_train, dist_features_test


def get_categorical_features(train, test, feature):
    unique_vals = list(set(train[feature].unique().tolist() 
                           + test[feature].unique().tolist()))
    feat_dict = {i + 1: e for i, e in enumerate(unique_vals)}
    feat_dict_reverse = {v: k for k, v in feat_dict.items()}

    train_feat = train[feature].apply(lambda x: feat_dict_reverse[x]).values
    test_feat = test[feature].apply(lambda x: feat_dict_reverse[x]).values

    return train_feat, test_feat, feat_dict, feat_dict_reverse
