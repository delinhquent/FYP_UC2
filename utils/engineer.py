from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import nltk
import numpy as np

import pandas as pd

from utils.clean import clean_text


def engineer_reviews(df, contractions_path, slangs_path):
    # clean verified purchase column
    df.loc[df.cleaned_verified != 1, 'cleaned_verified'] = 0
    
    # clean cleaned_text column
    df['cleaned_text'] = temp_new_text(list(df['cleaned_text']))
    
    # engineer word count feature
    df['word_count'] = df['decoded_comment'].str.split().str.len()
    df['word_count'] = df['word_count'].fillna(value=0)

    # engineer sample reviews
    df['sample_review'] = check_sample_text(df['cleaned_text'])

    # engineer incentivized reviews
    df['incentivized_review'] = [0] * len(df)
    df['incentivized_review'] = check_incentivized_text(df['cleaned_text'], contractions_path, slangs_path, df['incentivized_review'])

    # return dataframe
    return df

def check_incentivized_text(text, contractions_path, slangs_path, incentivized_review):
    incentivized_texts = ['I have received this product for a discount in exchange for my honest review.',\
                          'This product was received at no cost for review and inspection purposes.',\
                          'Just received and am completing to receive a free bottle. Will follow up after first month.']
                          
    cleaned_incentivized_text = clean_text(incentivized_texts, contractions_path, slangs_path)

    print("Fuzzy Matching for Incentivized Reviews...")
    match = list(map(lambda x: process.extractOne(x, cleaned_incentivized_text, scorer=fuzz.token_set_ratio,processor=lambda x: x), text))
    print("Fuzzy Matching Completed...")

    match_df = pd.DataFrame(match,columns=['incentivized_text', 'score'])
    match_df['probability_incentivized_text'] = match_df['score']/ (102 - match_df['score'])
    match_df['incentivized_review'] = incentivized_review

    match_df.loc[match_df.probability_incentivized_text >= 1, "incentivized_review"] = 1

    return list(match_df['incentivized_review'])

def check_sample_text(text):
    return [1 if 'sample' in data else 0 for data in text]

def temp_new_text(text):
    new_text = []
    for data in text:
        new_text.append(check_null_text(data))
    return new_text

def check_null_text(text):
    try:
        null_data = float(text)
        if math.isnan(null_data):
            return ""
        else:
            return str(text)
    except:
        return str(text)