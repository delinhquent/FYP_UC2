import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

import pandas as pd

from utils.clean import clean_text
from utils.engineer_functions import *


def engineer_reviews(df, sample_incentivized_list, products_df):
    # clean verified purchase column
    df.loc[df.cleaned_verified != 1, 'cleaned_verified'] = 0
    
    # clean cleaned_text column
    df['cleaned_text'] = temp_new_text(list(df['cleaned_text']))
    products_df['cleaned_text'] = temp_new_text(list(products_df['cleaned_text']))
    
    # engineer word count feature
    df['cleaned_word_count'] = df['decoded_comment'].str.split().str.len()
    df['cleaned_word_count'] = df['cleaned_word_count'].fillna(value=0)

    # engineer sample reviews
    df['cleaned_sample_review'] = check_sample_text(temp_new_text(df['decoded_comment']))

    # engineer incentivized reviews
    df['cleaned_incentivized_review'] = fuzzy_check_reviews(df['cleaned_text'], sample_incentivized_list, 'incentivized')

    # engineer deleted reviews
    df['cleaned_deleted_review'] = check_deleted_text(list(df['decoded_comment']))

    # engineer word count feature
    df['cleaned_word_count'] = df['decoded_comment'].str.split().str.len()
    df['cleaned_word_count'] = df['cleaned_word_count'].fillna(value=0)

    # engineer sentiment analysis for decoded and cleaned text
    sid = SentimentIntensityAnalyzer()
    df['cleaned_sentiment'] = sentiment_analysis(sid, list(df['decoded_comment'].astype(str)))

    # tfidf
    df = cosine_similarity(df, products_df)

    return df

def engineer_review_activity(df, loreal_brand_list, sample_incentivized_list):
    # engineer deleted reviews
    df['cleaned_deleted_review'] = check_deleted_text(list(df['decoded_comment']))
    
    # engineer word count feature
    df['cleaned_word_count'] = df['decoded_comment'].str.split().str.len()
    df['cleaned_word_count'] = df['cleaned_word_count'].fillna(value=0)
    
    # engineer sample reviews
    df['cleaned_sample_review'] = check_sample_text(temp_new_text(df['decoded_comment']))

    # engineer incentivized reviews
    df['cleaned_incentivized_review'] = fuzzy_check_reviews(temp_new_text(df['cleaned_text']), sample_incentivized_list,'incentivized')

    # engineer loreal reviews
    df['cleaned_loreal_review'] = fuzzy_check_reviews(temp_new_text(df['cleaned_text']), sample_incentivized_list, 'loreal')

    # clean cleaned_text column
    df['cleaned_text'] = temp_new_text(list(df['cleaned_text']))

    # clean verified purchase column
    df['cleaned_verified'] = 0 * len(df)
    df.loc[df.cleaned_verified == True, 'cleaned_verified'] = 1

    # clean date time posted column
    df['cleaned_datetime_posted'] = pd.to_datetime(df['cleaned_datetime_posted'], errors='coerce')
    df['cleaned_datetime_posted'] = df['cleaned_datetime_posted'].dt.date

    return df

def engineer_profiles(df, review_activity_df):
    # engineer reviewer ease and helpful Votes feature
    df = calculate_average(df, review_activity_df, 'cleaned_ratings', 'cleaned_reviewer_ease_score')
    df = calculate_average(df, review_activity_df, 'helpfulVotes', 'cleaned_average_helpfulVotes')

    # engineer total reviews posted
    df = calculate_total_reviews(df, review_activity_df, 'acc_num')

    # engineer total and proportion of reviews posted for loreal, deleted, verified, incentivized, sample, same day
    for column in ['cleaned_loreal_review','cleaned_incentivized_review','cleaned_verified','cleaned_sample_review']:
        df = total_proportion_reviews(df, review_activity_df, 'acc_num', column)
    df = deleted_reviews(df, review_activity_df, 'acc_num')
    df = profiles_same_day_reviews(df, review_activity_df, 'acc_num')

    # engineer average word length
    df = calculate_average_word_count(df, review_activity_df, 'acc_num')

    # engineer brand repeats class
    df = profiles_products_reviewed(df, review_activity_df)
    df = profiles_same_day_reviewer(df, review_activity_df)
    df = profiles_brand_repeats(df)

    # engineer suspicious class
    df = profiles_suspicious(df)

    # clean badges column
    df = clean_badges(df)

    # clean ranking column
    df['cleaned_ranking'] = df['ranking'].fillna(value=0)
    df['cleaned_ranking'] = df['cleaned_ranking'].astype(str).apply(lambda x: x.replace(',','')).astype(int)

    # clean deleted status column
    df.loc[df.cleaned_deleted_status == True, 'cleaned_deleted_status'] = 1
    df.loc[df.cleaned_deleted_status == False, 'cleaned_deleted_status'] = 0

    # return dataframe
    return df

def engineer_products(df,profiles_df,reviews_df):
    # rename columns for convenience
    reviews_df = reviews_df.rename(columns={"ASIN":"asin"})

    # engineer average review length for products
    df = calculate_average_word_count(df, reviews_df, 'asin')

    # engineer total reviews posted
    df = calculate_total_reviews(df, reviews_df, 'asin')

    # engineer total and proportion of reviews posted for loreal, deleted, verified, incentivized, sample
    for column in ['cleaned_incentivized_review','cleaned_verified','cleaned_sample_review']:
        df = total_proportion_reviews(df, reviews_df, 'asin', column)
    df = deleted_reviews(df, reviews_df, 'asin')

    # clean ratings column
    df = fill_empty_ratings(df, reviews_df)

    # engineer total and proportion of brand repeats, brand monogomist, brand loyalist, single day reviewers
    df = total_proportion_suspicious_brand_repeats(df, reviews_df, profiles_df)

    # return dataframe
    return df

def generate_modelling_dataset(reviews_df, profiles_df, products_df):
    print("Extracting columns from Reviews Dataset...")
    reviews_uninterested_columns = ['cleaned_location','cleaned_title']
    reviews_interested_columns = retrieve_interested_columns(reviews_df, reviews_uninterested_columns)
    reviews_df = reviews_df[['ASIN','acc_num'] + reviews_interested_columns + ['manual_label']]
    reviews_df = reviews_df.rename(columns={'ASIN':"asin"})
    reviews_df = rename_columns(reviews_df, reviews_interested_columns, '_reviews_')

    print("Extracting columns from Products Dataset...")
    products_uninterested_columns = ['cleaned_text']
    products_interested_columns = retrieve_interested_columns(products_df, products_uninterested_columns)
    products_df = products_df[['asin'] + products_interested_columns]
    products_df = rename_columns(products_df, products_interested_columns, '_products_')

    print("Extracting columns from Profiles Dataset...")
    print("Size of Profile Dataset before merging: {}...".format(profiles_df.shape))
    profiles_interested_columns = [column for column in profiles_df.columns if 'cleaned' in column]
    profiles_df = profiles_df[['acc_num'] + profiles_interested_columns]
    profiles_df = rename_columns(profiles_df, profiles_interested_columns, '_profiles_')
    profiles_df = profiles_df.drop_duplicates(subset=['acc_num'], keep='first')
    print("Size of Profile Dataset after merging: {}...".format(profiles_df.shape))

    print("Combining all columns into a single Dataset for modelling...")

    print("Size of Modelling Dataset before merging: {}...".format(reviews_df.shape))
    df = pd.merge(reviews_df,products_df,left_on=['asin'], right_on = ['asin'], how = 'left')

    df = pd.merge(df,profiles_df,left_on=['acc_num'], right_on = ['acc_num'], how = 'left')

    print("Size of Modelling Dataset after merging: {}...".format(df.shape))
    
    print("Size of Modelling Dataset after dropping null asin: {}...".format(df.shape))
    
    return df

