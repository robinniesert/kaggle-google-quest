import os
import numpy as np
import pandas as pd
import pickle
import gensim
import spacy

from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer


def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path,'rb') as f:
        emb_arr = pickle.load(f)
    return emb_arr


def build_matrix_adv(embedding_path, embedding_path_spellcheck, word_dict=None, 
                     lemma_dict=None, max_features=100000, embed_size=300,
                     matrix_path=''):
    nb_words = min(max_features, len(word_dict))
    if matrix_path != '':
        os.makedirs(matrix_path, exist_ok=True)
        embedding_name = embedding_path.split('/')[-1].split('.')[0]
        embedding_name += '_spellcheck_'
        embedding_name += embedding_path_spellcheck.split('/')[-1].split('.')[0]
        matrix_path += f'{embedding_name}_matrix.csv'

    if os.path.isfile(matrix_path):
        unknown_words = None
        embedding_matrix = pd.read_csv(matrix_path, index_col=0).values

    else:
        spell_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path_spellcheck)
        words = spell_model.index2word
        w_rank = {}
        for i, word in enumerate(words):
            w_rank[word] = i
        WORDS = w_rank

        def P(word):
            "Probability of `word`."
            # use inverse of rank as proxy
            # returns 0 if the word isn't in the dictionary
            return - WORDS.get(word, 0)

        def correction(word):
            "Most probable spelling correction for word."
            return max(candidates(word), key=P)

        def candidates(word):
            "Generate possible spelling corrections for word."
            return (known([word]) or known(edits1(word)) or [word])

        def known(words):
            "The subset of `words` that appear in the dictionary of WORDS."
            return set(w for w in words if w in WORDS)

        def edits1(word):
            "All edits that are one edit away from `word`."
            letters = 'abcdefghijklmnopqrstuvwxyz'
            splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
            deletes = [L + R[1:] for L, R in splits if R]
            transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
            replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
            inserts = [L + c + R for L, R in splits for c in letters]
            return set(deletes + transposes + replaces + inserts)

        ps = PorterStemmer()
        lc = LancasterStemmer()
        sb = SnowballStemmer("english")

        # embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding='utf-8'))
        # embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding='utf-8', errors='ignore'))
        embedding_index = load_embeddings(embedding_path)

        embedding_matrix = np.zeros((nb_words + 1, embed_size))
        unknown_words = []
        for word, i in word_dict.items():
            key = word
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                continue
            embedding_vector = embedding_index.get(word.lower())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                continue
            embedding_vector = embedding_index.get(word.upper())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                continue
            embedding_vector = embedding_index.get(word.capitalize())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                continue
            word = ps.stem(key)
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[word_dict[key]] = embedding_vector
                continue
            word = lc.stem(key)
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[word_dict[key]] = embedding_vector
                continue
            word = sb.stem(key)
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[word_dict[key]] = embedding_vector
                continue
            word = lemma_dict[key]
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[word_dict[key]] = embedding_vector
                continue
            if len(key) > 1:
                word = correction(key)
                embedding_vector = embedding_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[word_dict[key]] = embedding_vector
                    continue
            unknown_words.append(key)

        print(f'{len(unknown_words) * 100 / len(word_dict):.4f}% words are not in embeddings')

        if matrix_path != '':
            pd.DataFrame(embedding_matrix).to_csv(matrix_path)

    return embedding_matrix, nb_words, unknown_words


def get_word_lemma_dict(full_text, path=''):
    if path != '': os.makedirs(path, exist_ok=True)

    if os.path.isfile(path+'lemma_dict.pkl') and os.path.isfile(path+'word_dict.pkl'):
        lemma_dict = pickle.load(open(path+'lemma_dict.pkl', 'rb'))
        word_dict = pickle.load(open(path+'word_dict.pkl', 'rb'))
    else:
        nlp = spacy.load('en_core_web_lg', disable=['parser','ner','tagger'])
        nlp.vocab.add_flag(
            lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, 
            spacy.attrs.IS_STOP
        )
        word_dict = {}
        word_index = 1
        lemma_dict = {}
        docs = nlp.pipe(full_text, n_threads=os.cpu_count())
        for doc in docs:
            for token in doc:
                if (token.text not in word_dict) and (token.pos_ is not "PUNCT"):
                    word_dict[token.text] = word_index
                    word_index += 1
                    lemma_dict[token.text] = token.lemma_

        if path != '':
            pickle.dump(lemma_dict, open(path+'lemma_dict.pkl', 'wb'))
            pickle.dump(word_dict, open(path+'word_dict.pkl', 'wb'))

    return lemma_dict, word_dict