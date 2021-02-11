import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
import tqdm

import numpy as np

def create_embedding_df(reviews, model):
    vec = CountVectorizer(ngram_range = (1,1))
    vec.fit_transform(reviews)
    embedding_df = pd.DataFrame(columns=vec.get_feature_names())

    for i in tqdm.tqdm(range(len(reviews))):
        words = reviews[i].split()
        for word in words:
            embedding_df.at[i, word] = get_vector_value(word, model)
                
    return embedding_df.fillna(value=0)

def get_vector_value(word,model):
    try:
        vector_value = np.mean(model.wv[word])
    except Exception as e:
        vector_value = 0
    return vector_value

def avg_sentence_vector(words, model, num_features, vocab):
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in vocab:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])

    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec