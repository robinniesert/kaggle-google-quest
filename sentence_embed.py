import os
import numpy as np
import pandas as pd

import tensorflow_hub as hub
import tensorflow.keras.backend as K


def get_use_embedding_features(train, test, input_columns, use_feature_path=''):
    """
    https://www.kaggle.com/ragnar123/simple-lgbm-solution-baseline?scriptVersionId=24198335
    """


    # create empty dictionaries to store final results
    embedding_train = {}
    embedding_test = {}

    if os.path.isdir(use_feature_path):
        for text in input_columns:
            key = text + '_embedding'
            embedding_train[key] = pd.read_csv(
                f'{use_feature_path}{key}_train.csv', index_col=0).values
            embedding_test[key] = pd.read_csv(
                f'{use_feature_path}{key}_test.csv', index_col=0).values

    else:
        if use_feature_path != '': os.makedirs(use_feature_path, exist_ok=True)

        # load universal sentence encoder model to get sentence ambeddings
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/4"
        embed = hub.load(module_url)

        # iterate over text columns to get senteces embeddings with the previous 
        # loaded model
        for text in input_columns:
            print(text)
            train_text = train[text].str.replace('?', '.').str.replace('!', '.').tolist()
            test_text = test[text].str.replace('?', '.').str.replace('!', '.').tolist()
        
            # create empy list to save each batch
            curr_train_emb = []
            curr_test_emb = []
        
            # define a batch to transform senteces to their correspinding embedding 
            # (1 X 512 for each sentece)
            batch_size = 4
            ind = 0
            while ind * batch_size < len(train_text):
                s, e = ind * batch_size, (ind + 1) * batch_size
                curr_train_emb.append(embed(train_text[s:e])['outputs'].numpy())
                ind += 1
            
            ind = 0
            while ind * batch_size < len(test_text):
                s, e = ind * batch_size, (ind + 1) * batch_size
                curr_test_emb.append(embed(test_text[s:e])['outputs'].numpy())
                ind += 1

            # stack arrays to get a 2D array (dataframe) corresponding with all the 
            # sentences and dim 512 for columns (sentence encoder output)
            key = text + '_embedding'
            embedding_train[key] = np.vstack(curr_train_emb)
            embedding_test[key] = np.vstack(curr_test_emb)

            if use_feature_path != '':
                pd.DataFrame(embedding_train[key]).to_csv(
                    f'{use_feature_path}{text}_embedding_train.csv')
                pd.DataFrame(embedding_test[key]).to_csv(
                    f'{use_feature_path}{text}_embedding_test.csv')
        
        K.clear_session()
        
    return embedding_train, embedding_test