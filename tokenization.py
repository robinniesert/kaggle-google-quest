import numpy as np
from tqdm import tqdm

from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, XLNetTokenizer


tokenizers = {
    'bert-base-uncased': BertTokenizer,
    'roberta-base': RobertaTokenizer,
    'xlnet-base-cased': XLNetTokenizer,
    'albert-base-v2': AlbertTokenizer
}


def tokenize(df, pretrained_model_str='bert-base-uncased'):
    print(f'Tokenize inputs for model {pretrained_model_str}...')

    tokenizer = tokenizers[pretrained_model_str].from_pretrained(pretrained_model_str)
    seg_ids_all, ids_all = {}, {}
    max_seq_len = 512
    
    for text, cols in [('question', ['question_title', 'question_body']), 
                       ('answer', ['question_title', 'answer'])]:
        ids, seg_ids = [], []
        for x1, x2 in tqdm(df[cols].values):
            encoded_inputs = tokenizer.encode_plus(
                x1, x2, add_special_tokens=True, max_length=max_seq_len, 
                pad_to_max_length=True, return_token_type_ids=True
            )
            ids.append(encoded_inputs['input_ids'])
            seg_ids.append(encoded_inputs['token_type_ids'])
        
        ids_all[text] = np.array(ids)
        seg_ids_all[text] = np.array(seg_ids)
    
    return ids_all, seg_ids_all
