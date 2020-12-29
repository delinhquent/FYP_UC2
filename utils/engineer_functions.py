from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

import pandas as pd

from utils.clean import clean_text

def calculate_reviewer_ease(df, review_activity_df):
    reviewer_ease_df = review_activity_df.groupby('acc_num').agg({'cleaned_ratings':np.mean}).reset_index()
    reviewer_ease_df = reviewer_ease_df.rename(columns={'cleaned_ratings':'cleaned_reviewer_ease_score'})

    df = pd.merge(df,reviewer_ease_df,left_on=['acc_num'], right_on = ['acc_num'], how = 'left')
    df['cleaned_reviewer_ease_score'] = df['cleaned_reviewer_ease_score'].fillna(value=0)

    return df

def calculate_total_reviews(df, review_activity_df, column):
    total_products_df = review_activity_df.groupby(column).size().reset_index(name='cleaned_total_reviews_posted')

    df = pd.merge(df,total_products_df,left_on=[column], right_on = [column], how = 'left')
    df['cleaned_total_reviews_posted'] = df['cleaned_total_reviews_posted'].fillna(value=0)

    return df

def calculate_average_helpfulVotes(df, review_activity_df):
    helpfulVotes_df = review_activity_df.groupby('acc_num').agg({'helpfulVotes':np.mean}).reset_index()
    helpfulVotes_df = helpfulVotes_df.rename(columns={'helpfulVotes':'cleaned_average_helpfulVotes'})

    df = pd.merge(df,helpfulVotes_df,left_on=['acc_num'], right_on = ['acc_num'], how = 'left')
    df['cleaned_average_helpfulVotes'] = df['cleaned_average_helpfulVotes'].fillna(value=0)

    return df

def calculate_average_word_count(df, review_activity_df, column):
    review_length_df = review_activity_df.groupby(column).agg({'cleaned_word_count':np.mean}).reset_index()
    review_length_df = review_length_df.rename(columns={'cleaned_word_count':'cleaned_average_review_length'})

    df = pd.merge(df,review_length_df,left_on=[column], right_on = [column], how = 'left')
    df['cleaned_average_review_length'] = df['cleaned_average_review_length'].fillna(value=0)

    return df

def loreal_reviews(df, review_activity_df,column):
    loreal_products_df = review_activity_df[review_activity_df['cleaned_loreal_review'] == 1].groupby(column).size().reset_index(name='cleaned_total_loreal_reviews_posted')

    df = pd.merge(df,loreal_products_df,left_on=[column], right_on = [column], how = 'left')
    df['cleaned_total_loreal_reviews_posted'] = df['cleaned_total_loreal_reviews_posted'].fillna(value=0)

    df['cleaned_proportion_loreal_reviews_posted'] = df['cleaned_total_loreal_reviews_posted'] / df['cleaned_total_reviews_posted']
    df['cleaned_proportion_loreal_reviews_posted'] = df['cleaned_proportion_loreal_reviews_posted'].fillna(value=0)

    return df

def deleted_reviews(df, review_activity_df,column):
    total_deleted_reviews_df = review_activity_df.groupby([column]).agg({'cleaned_deleted_review':np.sum}).reset_index()
    total_deleted_reviews_df = total_deleted_reviews_df.rename(columns={'cleaned_deleted_review':'cleaned_total_deleted_reviews'})

    df = pd.merge(df,total_deleted_reviews_df,left_on=[column], right_on = [column], how = 'left')
    df['cleaned_total_deleted_reviews'] = df['cleaned_total_deleted_reviews'].fillna(value=0)

    df['cleaned_proportion_deleted_reviews_posted'] = df['cleaned_total_deleted_reviews'] / df['cleaned_total_reviews_posted']
    df['cleaned_proportion_deleted_reviews_posted'] = df['cleaned_proportion_deleted_reviews_posted'].fillna(value=0)

    return df

def verified_reviews(df, review_activity_df,column):
    verified_purchase_df = review_activity_df[review_activity_df['cleaned_verified'] == 1].groupby(column).size().reset_index(name='cleaned_total_verified_reviews')

    df = pd.merge(df,verified_purchase_df,left_on=[column], right_on = [column], how = 'left')
    df['cleaned_total_verified_reviews'] = df['cleaned_total_verified_reviews'].fillna(value=0)

    df['cleaned_proportion_verified_reviews_posted'] = df['cleaned_total_verified_reviews'] / df['cleaned_total_reviews_posted']
    df['cleaned_proportion_verified_reviews_posted'] = df['cleaned_proportion_verified_reviews_posted'].fillna(value=0)

    return df

def incentivized_reviews(df, review_activity_df,column):
    incentivized_df = review_activity_df[review_activity_df['cleaned_incentivized_review'] == 1].groupby(column).size().reset_index(name='cleaned_total_incentivized_reviews')

    df = pd.merge(df,incentivized_df,left_on=[column], right_on = [column], how = 'left')
    df['cleaned_total_incentivized_reviews'] = df['cleaned_total_incentivized_reviews'].fillna(value=0)

    df['cleaned_proportion_incentivized_reviews_posted'] = df['cleaned_total_incentivized_reviews'] / df['cleaned_total_reviews_posted']
    df['cleaned_proportion_incentivized_reviews_posted'] = df['cleaned_proportion_incentivized_reviews_posted'].fillna(value=0)

    return df

def sample_reviews(df, review_activity_df,column):
    sample_df = review_activity_df[review_activity_df['cleaned_sample_review'] == 1].groupby(column).size().reset_index(name='cleaned_total_sample_reviews')

    df = pd.merge(df,sample_df,left_on=[column], right_on = [column], how = 'left')
    df['cleaned_total_sample_reviews'] = df['cleaned_total_sample_reviews'].fillna(value=0)

    df['cleaned_proportion_sample_reviews_posted'] = df['cleaned_total_sample_reviews'] / df['cleaned_total_reviews_posted']
    df['cleaned_proportion_sample_reviews_posted'] = df['cleaned_proportion_sample_reviews_posted'].fillna(value=0)

    return df

def profiles_same_day_reviews(df, review_activity_df,column):
    datetime_review_activity_df = review_activity_df.groupby(['cleaned_datetime_posted',column]).size().reset_index(name='cleaned_total_same_day_reviews')

    df = pd.merge(df,datetime_review_activity_df[[column,'cleaned_total_same_day_reviews']],left_on=[column], right_on = [column], how = 'left')
    df['cleaned_total_same_day_reviews'] = df['cleaned_total_same_day_reviews'].fillna(value=0)

    df['cleaned_proportion_same_day_reviews_posted'] = df['cleaned_total_same_day_reviews'] / df['cleaned_total_reviews_posted']
    df['cleaned_proportion_same_day_reviews_posted'] = df['cleaned_proportion_same_day_reviews_posted'].fillna(value=0)

    return df

def profiles_products_reviewed(df, review_activity_df):
    total_products_df = review_activity_df.groupby(['acc_num']).agg({"asin":"nunique"}).reset_index()
    total_products_df = total_products_df.rename(columns={"asin":"cleaned_total_product"})

    loreal_products_df = review_activity_df[review_activity_df['cleaned_loreal_review'] == 1].groupby(['acc_num']).agg({"asin":"nunique"}).reset_index()
    loreal_products_df = loreal_products_df.rename(columns={"asin":"cleaned_total_loreal_product"})

    final_products_df = pd.merge(total_products_df,loreal_products_df,left_on=['acc_num'],right_on=['acc_num'],how='left')
    df = pd.merge(df,final_products_df,left_on=['acc_num'],right_on=['acc_num'],how='left')
    for column in ['cleaned_total_product','cleaned_total_loreal_product']:
        df[column] = df[column].fillna(value=0)
    df['cleaned_proportion_loreal_product'] = df['cleaned_total_loreal_product'] / df['cleaned_total_product']
    df['cleaned_proportion_loreal_product'] = df['cleaned_proportion_loreal_product'].fillna(value=0)
    return df

def profiles_brand_repeats(df):
    brand_monogamist = []
    brand_loyalist = []
    brand_repeater = []
    for index,row in df.iterrows():
        if row['cleaned_total_loreal_reviews_posted'] > 1 and row['cleaned_proportion_loreal_product'] == 1 and row['cleaned_proportion_loreal_reviews_posted'] == 1:
            brand_monogamist.append(1)
        else:
            brand_monogamist.append(0)
        if row['cleaned_total_reviews_posted'] > 1 and row['cleaned_proportion_loreal_product'] >= 0.5:
            brand_loyalist.append(1)
        else:
            brand_loyalist.append(0)
        if row['cleaned_proportion_loreal_product'] == 1:
            brand_repeater.append(1)
        else:
            brand_repeater.append(0)
    df['cleaned_brand_monogamist'] = brand_monogamist
    df['cleaned_brand_loyalist'] = brand_loyalist
    df['cleaned_brand_repeater'] = brand_repeater
    return df

def profiles_same_day_reviewer(df, review_activity_df):
    datetime_review_activity_df = review_activity_df.groupby([review_activity_df['cleaned_datetime_posted'],'acc_num']).size().reset_index(name='count')
    single_day_reviewers = list(set(datetime_review_activity_df[datetime_review_activity_df['count'] > 1]['acc_num']))
    df['cleaned_single_day_reviewer'] = 0

    df.loc[df['acc_num'].isin(single_day_reviewers), 'cleaned_single_day_reviewer'] = 1
    return df

def profiles_suspicious(df):
    df['cleaned_never_verified_reviewer'] = 0
    df.loc[(df['cleaned_total_verified_reviews'] == 0) & (df['cleaned_deleted_status'] == False),"cleaned_never_verified_reviewer"] = 1   

    df['cleaned_one_hit_wonder'] = 0
    df.loc[(df['cleaned_total_reviews_posted'] == 1) & (df['cleaned_deleted_status'] == False),"cleaned_one_hit_wonder"] = 1

    df['cleaned_take_back_reviewer'] = 0
    df.loc[(df['cleaned_total_deleted_reviews'] > 0) & (df['cleaned_deleted_status'] == False),"cleaned_take_back_reviewer"] = 1

    return df

def clean_badges(df):
    df['cleaned_badges'] = 0
    df.loc[df['badges'].notnull(),"cleaned_badges"] = 1

    return df

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

def check_null_text(text):
    try:
        null_data = float(text)
        if math.isnan(null_data):
            return ""
        else:
            return str(text)
    except:
        return str(text)

def check_sample_text(text):
    return [1 if 'sample' in data else 0 for data in text]

def fuzzy_check_reviews(text, fuzzy_match_list, mode):
    print("Fuzzy Matching...")
    match = fuzzy_match_results(fuzzy_match_list,text)

    print("Fuzzy Matching Completed...")

    match_df = pd.DataFrame(match,columns = ['brand', 'score'])
    match_df['probability_brand'] = fuzzy_score(list((match_df['score'])))
    match_df['class_assignment'] = 0

    match_df.loc[match_df.probability_brand >= 1, "class_assignment"] = 1

    return list(match_df['class_assignment'])

def fuzzy_match_results(matching_list, text_list):
    return list(map(lambda x: process.extractOne(x, matching_list, scorer=fuzz.token_set_ratio,processor=lambda x: x), text_list))

def fuzzy_score(score_list):
    return [(score / (102-score)) for score in score_list]

def temp_new_text(text):
    new_text = []
    for data in text:
        new_text.append(check_null_text(data))
    return new_text

def fill_empty_ratings(df, reviews_df):
    products_without_rating = list(set(df[df['cleaned_rating'].isnull()]['asin']))
    average_rating_df = reviews_df[reviews_df['asin'].isin(products_without_rating)].groupby('asin').agg({"cleaned_ratings": np.mean}).reset_index()

    temp_df = df[['asin','cleaned_rating']].set_index("asin").cleaned_rating.fillna(average_rating_df.set_index("asin").cleaned_ratings).reset_index()
    temp_df['cleaned_rating'] = temp_df['cleaned_rating'].fillna(value=0)

    df = df.drop(columns=['cleaned_rating'])
    df = pd.merge(df,temp_df,left_on=['asin'], right_on = ['asin'], how = 'left')
    df['cleaned_rating'] = df['cleaned_rating'].fillna(value=0)

    return df

def total_proportion_suspicious_brand_repeats(df, reviews_df, profiles_df):
    total_users_posted_df = reviews_df.groupby('asin').size().reset_index(name='cleaned_total_users_posted')

    interested_columns = ['cleaned_single_day_reviewer','cleaned_never_verified_reviewer','cleaned_one_hit_wonder','cleaned_take_back_reviewer','cleaned_brand_monogamist','cleaned_brand_loyalist','cleaned_brand_repeater']
    temp_df = pd.merge(reviews_df,profiles_df[['acc_num'] + interested_columns],left_on=['acc_num'], right_on = ['acc_num'], how = 'left')

    new_columns = []
    for column in interested_columns:
        column_name_list = column.split("_")
        total_column_name = column_name_list[0] + "_total_" + '_'.join(column_name_list[1:])
        proportion_column_name = column_name_list[0] + "_proportion_" + '_'.join(column_name_list[1:])
        new_columns += [total_column_name, proportion_column_name]

        current_df = temp_df[temp_df[column] == 1]
        current_df = current_df.groupby(['asin',column]).size().reset_index(name=total_column_name)

        current_df = pd.merge(current_df,total_users_posted_df,left_on=['asin'], right_on = ['asin'], how = 'left')
        current_df[proportion_column_name] = current_df[total_column_name] / current_df['cleaned_total_users_posted']

        total_users_posted_df = pd.merge(total_users_posted_df,current_df[['asin',total_column_name,proportion_column_name]],left_on=['asin'], right_on = ['asin'], how = 'left')
    for column in new_columns:
        total_users_posted_df[column] = total_users_posted_df[column].fillna(value=0)
    
    df = pd.merge(df,total_users_posted_df,left_on=['asin'], right_on = ['asin'], how = 'left')

    return df

def sentiment_analysis(sid, text):
    results = []
    for data in text:
        results.append(sid.polarity_scores(str(data))["compound"])
    return results