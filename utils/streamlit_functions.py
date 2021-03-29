from gensim.models.doc2vec import Doc2Vec
import hashlib
import pandas as pd
import pickle
import random

from data_loader.data_loader import DataLoader
from utils.config import process_config
from utils.utils import get_args
from utils.engineer_functions import *
from utils.reformat import *
from sklearn.metrics import pairwise_kernels
from sklearn.preprocessing import normalize

import streamlit as st


def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

def check_valid_user(username,password):
    credential_df = pd.read_csv('credentials.csv')
    if username in list(credential_df['user']):
        hashedpassword = credential_df[credential_df['user'] == username.lower()]['password'][0]
        authenticate = check_hashes(password,hashedpassword)
        if authenticate:
            return True
    return False

@st.cache(allow_output_mutation=True)
def load_all_relevant_data():
    args = get_args()
    config = process_config(args.config)
    model_config = process_config(args.model_config)
    results_data_loader = DataLoader(model_config.ocsvm.results.save_data_path)
    products_data_loader = DataLoader(config.products.interim_data_path)
    suspicious_profiles_data_loader = DataLoader(config.profiles.save_data_path)
    doc2vec_data_loader = DataLoader(config.doc2vec.reviews_vector)

    results_data_loader.load_data()
    products_data_loader.load_data()
    suspicious_profiles_data_loader.load_data()
    doc2vec_data_loader.load_data()

    results_data = results_data_loader.get_data()
    products_data = products_data_loader.get_data()
    suspicious_profiles_data = suspicious_profiles_data_loader.get_data()
    doc2vec_data = doc2vec_data_loader.get_data()

    products_data['name'] = products_data['name'].str.replace("-"," ")

    rename_columns = [column for column in products_data.columns if 'cleaned_' in column]
    new_column_names = {}
    for column in rename_columns:
        column_name_list = column.split("_")
        new_column_names[column] = column_name_list[0] + '_products_' + '_'.join(column_name_list[1:])
    products_data = products_data.rename(columns=new_column_names)

    suspicious_profiles_data = suspicious_profiles_data[ (suspicious_profiles_data['acc_num'].notnull())]
    suspicious_profiles_data['proportion_fake_reviews'] = suspicious_profiles_data['proportion_fake_reviews'].fillna(0)
    suspicious_profiles_data['suspicious_reviewer_score'] = suspicious_profiles_data['suspicious_reviewer_score'].fillna(0)

    rename_columns = [column for column in suspicious_profiles_data.columns if 'cleaned_' in column]
    new_column_names = {}
    for column in rename_columns:
        column_name_list = column.split("_")
        new_column_names[column] = column_name_list[0] + '_profiles_' + '_'.join(column_name_list[1:])
    suspicious_profiles_data = suspicious_profiles_data.rename(columns=new_column_names)

    results_data['decision_function'] = [value if value != 1 else round(random.uniform(0.90000, 0.99999),5) for value in results_data['decision_function']]

    results_data = results_data[ (results_data['acc_num'].notnull()) & (results_data['cleaned_reviews_text'].notnull())]

    return products_data, suspicious_profiles_data, results_data, doc2vec_data

@st.cache(allow_output_mutation=True)
def load_all_relevant_models():
    args = get_args()
    config = process_config(args.config)
    with open('models/ocsvm.pkl', 'rb') as pickle_file:
        unnatural_reviewer_model = pickle.load(pickle_file)
    with open('models/normalizer/feature_normalizer_standard.pkl', 'rb') as pickle_file:
        standard_normalizer = pickle.load(pickle_file)
    doc2vec_model = Doc2Vec.load(config.doc2vec.model_file)
    with open('models/normalizer/cosine_similarity_tfidf.pkl', 'rb') as pickle_file:
        tfidf_normalizer = pickle.load(pickle_file)

    return unnatural_reviewer_model, standard_normalizer, doc2vec_model, tfidf_normalizer

def generate_profile_options(profiles_data):
    unique_reviewer_scores = list(set(profiles_data['suspicious_reviewer_score']))

    profile_options = []
    for reviewer_score in unique_reviewer_scores:
        current_df = profiles_data[profiles_data['suspicious_reviewer_score'] == reviewer_score]
        try:
            current_df['quantile'] = pd.qcut(current_df['proportion_fake_reviews'],5)
            print(current_df.columns)
            unique_bins = list(set(current_df['quantile']))
            for unique_bin in unique_bins:
                temp_df = current_df[(current_df['quantile'] == unique_bin) & (current_df['cleaned_total_reviews_posted'] >= 8)][:5]
                for suspicious_reviewer_score, proportion_fake_reviews, name in zip(temp_df['suspicious_reviewer_score'],temp_df['proportion_fake_reviews'],temp_df['name']):
                    new_score = str(suspicious_reviewer_score*100)
                    new_proportion_fake_reviews = str(proportion_fake_reviews*100)
                    profile_options.append("{} ({}% Suspicious - Posted {}% Unnatural Reviews)".format(name, new_score,new_proportion_fake_reviews))
        except:
            temp_df = current_df[:5]
            for suspicious_reviewer_score, proportion_fake_reviews, name in zip(temp_df['suspicious_reviewer_score'],temp_df['proportion_fake_reviews'],temp_df['name']):
                new_score = str(suspicious_reviewer_score*100)
                new_proportion_fake_reviews = str(proportion_fake_reviews*100)
                profile_options.append("{} ({}% Suspicious - Posted {}% Unnatural Reviews)".format(name, new_score,new_proportion_fake_reviews))

    return profile_options

def combine_product_profile_df(current_product_df, current_profile_df):

    test_df = pd.concat([current_product_df.reset_index().drop(columns='index'), current_profile_df.reset_index().drop(columns='index')], axis=1, ignore_index=True)
    test_df.columns = list(current_product_df.columns) + list(current_profile_df.columns)

    return test_df

def custom_feature_engineering(test_df, review_rating, review_verified, review_votes, review_text,tfidf_normalizer):
    args = get_args()
    config = process_config(args.config)
    test_df['cleaned_reviews_ratings'] = review_rating 
    if review_verified:
        new_review_verified = 1
    else:
        new_review_verified = 0
    test_df['cleaned_reviews_verified'] = new_review_verified
    test_df['cleaned_reviews_voting'] = review_votes
    test_df['cleaned_reviews_word_count'] = len(review_text.split())
    decoded_text = decode_comments(review_text).replace('\n', ' ').replace('\t', ' ').lower().strip()
    cleaned_text = clean_text([decoded_text], config.preprocessing.contractions_path, config.preprocessing.slangs_path)
    test_df['cleaned_reviews_sample_review'] = check_sample_text(temp_new_text([decoded_text]))[0]
    test_df['cleaned_reviews_incentivized_review'] = fuzzy_check_reviews(temp_new_text(cleaned_text), config.user_inputs.sample_incentivized_path, 'incentivized')[0]
    sid = SentimentIntensityAnalyzer()
    test_df['cleaned_reviews_sentiment'] = sentiment_analysis(sid, temp_new_text([decoded_text]))[0]
    test_df['cleaned_reviews_sentiment'] = test_df['cleaned_reviews_sentiment'].fillna(value=0)
    
    decoded_description = decode_comments(list(test_df['description'])[0]).replace('\n', ' ').replace('\t', ' ').lower().strip()
    cleaned_description = clean_text([decoded_description], config.preprocessing.contractions_path, config.preprocessing.slangs_path)
    similarity_score = float(list(pairwise_kernels(tfidf_normalizer.transform(cleaned_text),tfidf_normalizer.transform(cleaned_description), metric='cosine'))[0])
    test_df['cleaned_reviews_cosine_sim_product_detail'] = similarity_score
    test_df['cleaned_reviews_deleted_review'] = 0

    return test_df, cleaned_text

def custom_merge_doc2vec(test_df, doc2vec_values):
    columns = doc2vec_values.shape[0]
    
    doc2vec_columns = ['doc2vec_feature'+str(i) for i in range(1,columns+1)]
    for doc2vec_column, doc2vec_value in zip(doc2vec_columns, list(doc2vec_values)):
        test_df[doc2vec_column] = doc2vec_value
    
    return test_df.fillna(0)

def predict_custom_inputs(unnatural_reviewer_model, X_normalized, min_decision_function, max_decision_function):
    result = unnatural_reviewer_model.predict(X_normalized)[0]
    decision = unnatural_reviewer_model.decision_function(X_normalized)[0]
    if result == 0:
        result_text = "natural"
    else:
        result_text = "unnatural"
    if decision > max_decision_function:
        model_confidence = max_decision_function
    elif decision < min_decision_function:
        model_confidence = min_decision_function
    else:
        model_confidence = (float(decision) - min_decision_function) /(max_decision_function - min_decision_function)
    
    return result, result_text, round(model_confidence,5)

def evaluate_timescore(diff_days):
    if diff_days <= 30:
        return 1
    elif diff_days <= 90:
        return 0.8
    elif diff_days <= 180:
        return 0.6
    elif diff_days <= 360:
        return 0.4
    elif diff_days <= 720:
        return 0.2
    elif diff_days > 720:
        return 0

def evaluate_business_impact(max_helpful_votes, min_helpful_votes, review_votes, result, model_confidence, suspicious_reviewer_score, proportion_fake_reviews,diff_days):
    time_score = evaluate_timescore(diff_days)
    normalized_voting = (review_votes - min_helpful_votes)/(max_helpful_votes - min_helpful_votes)

    impact_score = (time_score + normalized_voting + (result*model_confidence) + suspicious_reviewer_score + proportion_fake_reviews)/5
    
    if impact_score >= 0.7:
        impact_text = "very severe"
    elif impact_score >=0.5:
        impact_text = "severe"
    else:
        impact_text = "not severe"
    
    return impact_score, impact_text