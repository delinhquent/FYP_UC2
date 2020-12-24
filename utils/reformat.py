import ast
import datetime
import html

import icu
import nltk
import numpy as np

import re

from unicodedata import normalize
from utils.clean import clean_text

resources = ["wordnet", "stopwords", "punkt", \
             "averaged_perceptron_tagger", "maxent_treebank_pos_tagger", "wordnet"]

for resource in resources:
    try:
        nltk.data.find("tokenizers/" + resource)
    except LookupError:
        nltk.download(resource)


def reformat_review_activity_df(df, contractions_path, slangs_path):
    print("Preprocessing title...")
    # Lowercase title text
    df['cleaned_title'] = df['title'].str.lower()

    print("Preprocessing ratings...")
    # Cleaning stars
    df['cleaned_ratings'] = df.rating.astype(str).apply(normalize_ratings)

    print("Preprocessing verified purchase...")
    # Data wrangling & Fill in blanks for Verified Puchase
    df['cleaned_verified'] = df.verifiedPurchase
    df.loc[df.cleaned_verified == True, 'cleaned_verified'] = 1
    df.loc[df.cleaned_verified == False, 'cleaned_verified'] = 0

    print("Preprocessing helpful votes...")
    # Data wrangling for voting columns
    df['cleaned_voting'] = df.helpfulVotes.astype(int)

    print("Preprocessing review count...")
    # Data wrangling for reviewCount columns
    df['cleaned_review_count'] = df.reviewCount.astype(int)

    print("Preprocessing number of image posted...")
    # Data wrangling for image_posted columns
    df['cleaned_images_posted'] = df.images_posted.astype(int)

    print("Preprocessing date time posted...")
    # convert integer to datetime for sortTimestamp
    df['cleaned_datetime_posted'] = df.sortTimestamp.apply(lambda x: datetime.datetime.fromtimestamp(x / 1000))

    print("Preprocessing reviews...")
    # Decode & lowercase comment text
    df['decoded_comment'] = df.text.astype(str).apply(decode_comments)
    df['decoded_comment'] = df['decoded_comment'].str.replace('\n', ' ').str.replace('\t', ' ').str.lower().str.strip()
    df['cleaned_text'] = clean_text(df['decoded_comment'], contractions_path, slangs_path)
    
    # return dataframe
    return df

def normalize_ratings(ratings):
    ratings += " out of 5"
    s_ratings = ratings.split()
    return float(s_ratings[0]) / float(s_ratings[3])


def decode_comments(text):
    text = normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    text = html.unescape(text)
    return ''.join([x for x in text if x.isprintable()])
