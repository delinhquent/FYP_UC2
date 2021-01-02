from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_kernels

from utils.clean import clean_text

def automatic_column_name(column,add_list):
    column_name_list = column.split("_")
    current_columns = []
    for current in add_list:
        current_columns.append(column_name_list[0] + current + '_'.join(column_name_list[1:]))
    return current_columns

def calculate_average(df, review_activity_df, column, new_column_name):
    temp_df = review_activity_df.groupby('acc_num').agg({column:np.mean}).reset_index()
    temp_df = temp_df.rename(columns={column:new_column_name})

    df = pd.merge(df,temp_df,left_on=['acc_num'], right_on = ['acc_num'], how = 'left')
    df[new_column_name] = df[new_column_name].fillna(value=0)

    return df

def calculate_total_reviews(df, review_activity_df, column):
    total_products_df = review_activity_df.groupby(column).size().reset_index(name='cleaned_total_reviews_posted')

    df = pd.merge(df,total_products_df,left_on=[column], right_on = [column], how = 'left')
    df['cleaned_total_reviews_posted'] = df['cleaned_total_reviews_posted'].fillna(value=0)

    return df

def calculate_average_word_count(df, review_activity_df, column):
    review_length_df = review_activity_df.groupby(column).agg({'cleaned_word_count':np.mean}).reset_index()
    review_length_df = review_length_df.rename(columns={'cleaned_word_count':'cleaned_average_review_length'})

    df = pd.merge(df,review_length_df,left_on=[column], right_on = [column], how = 'left')
    df['cleaned_average_review_length'] = df['cleaned_average_review_length'].fillna(value=0)

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

def cosine_similarity(df, products_df, tfidf_save_path):
    print("Conducting TFIDF...")
    vec = TfidfVectorizer (ngram_range = (1,2), max_features = 100)
    vec.fit(df['cleaned_text'])

    print("Saving TFIDF vector")
    tfidf = vec.transform(df['cleaned_text'])
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=vec.get_feature_names())
    tfidf_df.to_csv(tfidf_save_path, index=False)

    temp_df = df[['ASIN','cleaned_text']]
    temp_df = temp_df.rename(columns={'ASIN':'asin','cleaned_text':'cleaned_reviews_text'})
    temp_df = pd.merge(temp_df,products_df[['asin','cleaned_text']],left_on=['asin'], right_on = ['asin'], how = 'left')
    temp_df = temp_df.rename(columns={'cleaned_text':'cleaned_product_text'})

    print("Conducting Cosine Similarity...")
    review_text = temp_df['cleaned_reviews_text']
    product_detail_results = []
    for index, row in temp_df.iterrows():
        print("Cosine Similarity at {} of {}...".format(index+1, len(temp_df)))
        try:
            score = float(list(pairwise_kernels(vec.transform([row['cleaned_reviews_text']]),vec.transform([row['cleaned_product_text']]), metric='cosine'))[0])
            product_detail_results.append(score)
        except:
            product_detail_results.append(0)
    df['cleaned_cosine_sim_product_detail'] = product_detail_results

    return df

def deleted_reviews(df, review_activity_df,column):
    total_deleted_reviews_df = review_activity_df.groupby([column]).agg({'cleaned_deleted_review':np.sum}).reset_index()
    total_deleted_reviews_df = total_deleted_reviews_df.rename(columns={'cleaned_deleted_review':'cleaned_total_deleted_reviews'})

    df = pd.merge(df,total_deleted_reviews_df,left_on=[column], right_on = [column], how = 'left')
    df['cleaned_total_deleted_reviews'] = df['cleaned_total_deleted_reviews'].fillna(value=0)

    df['cleaned_proportion_deleted_reviews'] = df['cleaned_total_deleted_reviews'] / df['cleaned_total_reviews_posted']
    df['cleaned_proportion_deleted_reviews'] = df['cleaned_proportion_deleted_reviews'].fillna(value=0)

    return df

def divde_by_column(df, new_column, current_column, total_column):
    df[new_column] = df[current_column] / df[total_column]
    return df

def existing_with_extra_condition(df, condition, column,new_value):
    df.loc[(condition) & (df['cleaned_deleted_status'] == False), column] = new_value

def fill_empty_ratings(df, reviews_df):
    products_without_rating = list(set(df[df['cleaned_rating'].isnull()]['asin']))
    average_rating_df = reviews_df[reviews_df['asin'].isin(products_without_rating)].groupby('asin').agg({"cleaned_ratings": np.mean}).reset_index()

    temp_df = df[['asin','cleaned_rating']].set_index("asin").cleaned_rating.fillna(average_rating_df.set_index("asin").cleaned_ratings).reset_index()
    temp_df['cleaned_rating'] = temp_df['cleaned_rating'].fillna(value=0)

    df = df.drop(columns=['cleaned_rating'])
    df = pd.merge(df,temp_df,left_on=['asin'], right_on = ['asin'], how = 'left')
    df['cleaned_rating'] = df['cleaned_rating'].fillna(value=0)

    return df

def fill_empty_values(df, current_column,new_value):
    df[current_column] = df[current_column].fillna(value=new_value)
    return df

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

def profiles_brand_repeats(df):
    df['cleaned_brand_monogamist'] = 0
    df['cleaned_brand_loyalist'] = 0
    df['cleaned_brand_repeater'] = 0
    
    df.loc[(df['cleaned_total_loreal_product'] > 1) & (df['cleaned_proportion_loreal_product'] == 1) &  (df['cleaned_proportion_loreal_review'] == 1),"cleaned_brand_monogamist"] = 1 
    df.loc[(df['cleaned_total_reviews_posted'] > 1) & (df['cleaned_proportion_loreal_product'] >= 0.5),"cleaned_brand_loyalist"] = 1 
    df.loc[(df['cleaned_proportion_loreal_product'] == 1),"cleaned_brand_repeater"] = 1 

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

def profiles_same_day_reviews(df, review_activity_df,column):
    datetime_review_activity_df = review_activity_df.groupby(['cleaned_datetime_posted',column]).size().reset_index(name='cleaned_total_same_day_reviews')

    df = pd.merge(df,datetime_review_activity_df[[column,'cleaned_total_same_day_reviews']],left_on=[column], right_on = [column], how = 'left')
    df['cleaned_total_same_day_reviews'] = df['cleaned_total_same_day_reviews'].fillna(value=0)

    df['cleaned_proportion_same_day_reviews'] = df['cleaned_total_same_day_reviews'] / df['cleaned_total_reviews_posted']
    df['cleaned_proportion_same_day_reviews'] = df['cleaned_proportion_same_day_reviews'].fillna(value=0)

    return df

def profiles_same_day_reviewer(df, review_activity_df):
    datetime_review_activity_df = review_activity_df.groupby([review_activity_df['cleaned_datetime_posted'],'acc_num']).size().reset_index(name='count')
    single_day_reviewers = list(set(datetime_review_activity_df[datetime_review_activity_df['count'] > 1]['acc_num']))
    df['cleaned_single_day_reviewer'] = 0

    df.loc[df['acc_num'].isin(single_day_reviewers), 'cleaned_single_day_reviewer'] = 1
    return df

def profiles_suspicious(df):
    df['cleaned_never_verified_reviewer'] = 0
    existing_with_extra_condition(df, (df['cleaned_total_verified'] == 0), "cleaned_never_verified_reviewer",1)

    df['cleaned_one_hit_wonder'] = 0
    df.loc[(df['cleaned_total_reviews_posted'] == 1) & (df['cleaned_deleted_status'] == False),"cleaned_one_hit_wonder"] = 1

    df['cleaned_take_back_reviewer'] = 0
    existing_with_extra_condition(df, (df['cleaned_total_deleted_reviews'] > 0), "cleaned_take_back_reviewer",1)

    return df

def sentiment_analysis(sid, text):
    results = []
    for data in text:
        results.append(sid.polarity_scores(str(data))["compound"])
    return results

def rename_columns(df, columns, additional_word):
    for column in columns:
        df = df.rename(columns={column: automatic_column_name(column,[additional_word])[0]})
    return df

def temp_new_text(text):
    new_text = []
    for data in text:
        new_text.append(check_null_text(data))
    return new_text

def total_proportion_reviews(df, review_activity_df, groupby_column, column):
    current_columns = automatic_column_name(column,['_total_', '_proportion_'])
    temp_df = review_activity_df[review_activity_df[column] == 1].groupby(groupby_column).size().reset_index(name=current_columns[0])

    df = pd.merge(df,temp_df,left_on=[groupby_column], right_on = [groupby_column], how = 'left')
    df = fill_empty_values(df, current_columns[0],0)

    df = divde_by_column(df, current_columns[1], current_columns[0], 'cleaned_total_reviews_posted')
    df = fill_empty_values(df, current_columns[1],0)

    return df

def total_proportion_suspicious_brand_repeats(df, reviews_df, profiles_df):
    total_users_posted_df = reviews_df.groupby('asin').size().reset_index(name='cleaned_total_users_posted')

    interested_columns = ['cleaned_single_day_reviewer','cleaned_never_verified_reviewer','cleaned_one_hit_wonder','cleaned_take_back_reviewer','cleaned_brand_monogamist','cleaned_brand_loyalist','cleaned_brand_repeater']
    temp_df = pd.merge(reviews_df,profiles_df[['acc_num'] + interested_columns],left_on=['acc_num'], right_on = ['acc_num'], how = 'left')

    new_columns = []
    for column in interested_columns:
        current_columns = automatic_column_name(column,['_total_', '_proportion_'])
        new_columns += current_columns

        current_df = temp_df[temp_df[column] == 1]
        current_df = current_df.groupby(['asin',column]).size().reset_index(name=current_columns[0])

        current_df = pd.merge(current_df,total_users_posted_df,left_on=['asin'], right_on = ['asin'], how = 'left')
        current_df = divde_by_column(current_df, current_columns[1], current_columns[0], 'cleaned_total_users_posted')

        total_users_posted_df = pd.merge(total_users_posted_df,current_df[['asin',current_columns[0],current_columns[1]]],left_on=['asin'], right_on = ['asin'], how = 'left')
    for column in new_columns:
        total_users_posted_df[column] = total_users_posted_df[column].fillna(value=0)
    
    df = pd.merge(df,total_users_posted_df,left_on=['asin'], right_on = ['asin'], how = 'left')

    return df