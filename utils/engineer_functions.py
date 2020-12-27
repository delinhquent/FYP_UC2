from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import nltk
import numpy as np

import pandas as pd

from utils.clean import clean_text


def check_deleted_data(data):
    try:
        null_review = float(review)
        if math.isnan(null_review):
            return True
        return False
    except:
        return False

def check_deleted_text(text):
    new_data = []
    for data in text:
        if check_deleted_data(data):
            new_data.append(1)
        else:
            new_data.append(0)
    return new_data

def check_loreal_reviews(text, contractions_path,slangs_path, loreal_review):
    loreal_brands = ['loreal', 'garnier' 'maybelline','nyx','shu uemura','lancome', 'giorgio armani', \
                    'armani', 'yvessaintlaurent','kiehls','ralph lauren', 'urban decay', 'it cosmetic', \
                    'kerastase', 'vichy', 'la roche posay', 'skinceuticals']

    loreal_brands += clean_text(loreal_brands, contractions_path,slangs_path)

    print("Fuzzy Matching for Products under Loreal...")
    match = list(map(lambda x: process.extractOne(x, loreal_brands, scorer=fuzz.token_set_ratio, processor=lambda x: x), text))

    print("Fuzzy Matching Completed...")

    match_df = pd.DataFrame(match,columns = ['brand', 'score'])
    match_df['probability_brand'] = match_df['score']/(102 - match_df['score'])
    match_df['loreal_review'] = loreal_review

    match_df.loc[match_df.probability_brand >= 1, "loreal_review"] = 1

    return list(match_df['loreal_review'])

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