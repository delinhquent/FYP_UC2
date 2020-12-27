import nltk
import numpy as np

import pandas as pd

from utils.clean import clean_text
from utils.engineer_functions import *


def engineer_reviews(df, contractions_path, slangs_path):
    # clean verified purchase column
    df.loc[df.cleaned_verified != 1, 'cleaned_verified'] = 0
    
    # clean cleaned_text column
    df['cleaned_text'] = temp_new_text(list(df['cleaned_text']))
    
    # engineer word count feature
    df['cleaned_word_count'] = df['decoded_comment'].str.split().str.len()
    df['cleaned_word_count'] = df['cleaned_word_count'].fillna(value=0)

    # engineer sample reviews
    df['cleaned_sample_review'] = check_sample_text(temp_new_text(df['decoded_comment']))

    # engineer incentivized reviews
    df['cleaned_incentivized_review'] = [0] * len(df)
    df['cleaned_incentivized_review'] = fuzzy_check_reviews(df['cleaned_text'], contractions_path, slangs_path, df['cleaned_incentivized_review'], 'incentivized')

    # return dataframe
    return df

def engineer_review_activity(df, contractions_path, slangs_path):
    # engineer deleted reviews
    df['cleaned_deleted_review'] = check_deleted_text(list(df['decoded_comment']))
    
    # engineer word count feature
    df['cleaned_word_count'] = df['decoded_comment'].str.split().str.len()
    df['cleaned_word_count'] = df['cleaned_word_count'].fillna(value=0)
    
    # engineer sample reviews
    df['cleaned_sample_review'] = check_sample_text(temp_new_text(df['cleaned_text']))

    # engineer incentivized reviews
    df['cleaned_incentivized_review'] = [0] * len(df)
    df['cleaned_incentivized_review'] = fuzzy_check_reviews(temp_new_text(df['cleaned_text']), contractions_path, slangs_path, df['cleaned_incentivized_review'] ,'incentivized')

    # engineer loreal reviews
    df['cleaned_loreal_review'] = [0] * len(df)
    df['cleaned_loreal_review'] = fuzzy_check_reviews(temp_new_text(df['cleaned_text']), contractions_path, slangs_path, df['cleaned_loreal_review'], 'loreal')

    
    # clean cleaned_text column
    df['cleaned_text'] = temp_new_text(list(df['cleaned_text']))

    # return dataframe
    return df


