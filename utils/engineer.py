import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

import pandas as pd

from utils.clean import clean_text
from utils.engineer_functions import *


def engineer_reviews(df, sample_incentivized_list):
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
    df['cleaned_incentivized_review'] = fuzzy_check_reviews(df['cleaned_text'], sample_incentivized_list, 'incentivized')

    # engineer deleted reviews
    df['cleaned_deleted_review'] = check_deleted_text(list(df['decoded_comment']))

    # engineer sample reviews
    df['cleaned_sample_review'] = check_sample_text(temp_new_text(df['cleaned_text']))

    # engineer word count feature
    df['cleaned_word_count'] = df['decoded_comment'].str.split().str.len()
    df['cleaned_word_count'] = df['cleaned_word_count'].fillna(value=0)

    # engineer sentiment analysis for decoded and cleaned text
    sid = SentimentIntensityAnalyzer()
    df['cleaned_sentiment_decoded'] = sentiment_analysis(sid, list(df['decoded_comment'].astype(str)))
    df['cleaned_sentiment_cleaned'] = sentiment_analysis(sid, list(df['cleaned_text'].astype(str)))

    # return dataframe
    return df

def engineer_review_activity(df, loreal_brand_list, sample_incentivized_list):
    # engineer deleted reviews
    df['cleaned_deleted_review'] = check_deleted_text(list(df['decoded_comment']))
    
    # engineer word count feature
    df['cleaned_word_count'] = df['decoded_comment'].str.split().str.len()
    df['cleaned_word_count'] = df['cleaned_word_count'].fillna(value=0)
    
    # engineer sample reviews
    df['cleaned_sample_review'] = check_sample_text(temp_new_text(df['cleaned_text']))

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

    # return dataframe
    return df

def engineer_profiles(df, review_activity_df):
    # engineer reviewer ease feature
    df = calculate_reviewer_ease(df, review_activity_df)

    # engineer total reviews posted
    df = calculate_total_reviews(df, review_activity_df, 'acc_num')

    # engineer total and proportion of reviews posted for loreal, deleted, verified, incentivized, sample, same day
    df = loreal_reviews(df, review_activity_df, 'acc_num')
    df = deleted_reviews(df, review_activity_df, 'acc_num')
    df = verified_reviews(df, review_activity_df, 'acc_num')
    df = incentivized_reviews(df, review_activity_df, 'acc_num')
    df = sample_reviews(df, review_activity_df, 'acc_num')
    df = profiles_same_day_reviews(df, review_activity_df, 'acc_num')

    # engineer average word length
    df = calculate_average_word_count(df, review_activity_df, 'acc_num')

    # engineer average helpful votes
    df = calculate_average_helpfulVotes(df, review_activity_df)

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
    df = deleted_reviews(df, reviews_df, 'asin')
    df = verified_reviews(df, reviews_df, 'asin')
    df = incentivized_reviews(df, reviews_df, 'asin')
    df = sample_reviews(df, reviews_df, 'asin')

    # clean ratings column
    df = fill_empty_ratings(df, reviews_df)

    # engineer total and proportion of brand repeats, brand monogomist, brand loyalist, single day reviewers
    df = total_proportion_suspicious_brand_repeats(df, reviews_df, profiles_df)

    # return dataframe
    return df

