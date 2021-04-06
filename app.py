import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from SessionState import SessionState
from utils import common_resources
from utils.streamlit_functions import *
from PIL import Image
import scipy.stats as stats
import numpy as np


st.set_page_config(page_title='Unnatural Reviews Detector', layout = 'wide') # page_icon to put favicon, layout = "wide" or "centered"

session_state = SessionState.get(logged_in=False)

image = Image.open('images/custom_banner.png')
st.image(image)

placeholder_title = st.empty()
placeholder_username = st.empty()
placeholder_password = st.empty()
placeholder_button = st.empty()

placeholder_title.title('Login to Unnatural Reviews Detector')
user = placeholder_username.text_input('Username')
password = placeholder_password.text_input('Password',type='password')
if (placeholder_button.button("Login")):
    logged_in = check_valid_user(user,password)
    if logged_in:
        session_state.logged_in = True
    else:
        st.error("Invalid Username or Password")

if session_state.logged_in:
    placeholder_title.empty()
    placeholder_username.empty()
    placeholder_password.empty()
    placeholder_button.empty()

    st.title('Welcome to Unnatural Reviews Detector')
    st.warning("Changing any of the inputs might cause the script to re-run...While waiting, you can still adjust the inputs...")
    with st.spinner("Loading relevant data from database & models...Please wait..."):
        products_data, profiles_data, reviews_data, doc2vec_data, authentic_test_df, top10_features = load_all_relevant_data()
        unnatural_reviewer_model, standard_normalizer, doc2vec_model,tfidf_normalizer = load_all_relevant_models()
        profile_inverse_mapping = dict(zip(profiles_data['acc_num'], profiles_data['name']))   
        product_mapping = dict(zip(products_data['name'], products_data['asin']))
        profile_mapping = dict(zip(profiles_data['name'], profiles_data['acc_num']))

    with st.beta_container():
        product_options = list(products_data['name'])
        
        st.header("👨‍👩‍👧‍👦 Choose a Product & Profile...")
        product_option = st.selectbox('Select a Product:', product_options)
        current_asin = product_mapping[product_option]

        current_product_df = products_data[products_data['asin'] == current_asin]
        product_url = "https://www.amazon.com/dp/" + current_asin
        st.write("** Product's URL: **{}".format(product_url))

        profile_options = generate_profile_options(profiles_data)

        profile_option = st.selectbox("Select a Reviewer's Profile:", profile_options)
        name = profile_option.split(" (")[0]
        current_acc_num = profile_mapping[name]

        current_profile_df = profiles_data[profiles_data['acc_num'] == current_acc_num]
        product_url = "https://www.amazon.com/gp/profile/amzn1.account." + current_acc_num
        st.write("** Profile's URL: ** {}".format(product_url))
    
    st.write('')

    st.header("⭐ Customize Review's Input...")
    with st.beta_container():
        review_col1, review_col2 = st.beta_columns(2)
        with review_col1:
            min_date = datetime.datetime.today().date() - datetime.timedelta(days=(5*365)) 
            review_date = st.date_input('Please choose a date...', min_value=min_date, max_value=datetime.datetime.today().date())
            review_votes = st.number_input("Number of Helpful Votes?",0, max(reviews_data['cleaned_reviews_voting']))
        
        with review_col2:
            review_rating = st.slider('Please provide a rating...', 1, 5, 1)
            review_verified = st.checkbox("Verified Purchase?")

        review_text = st.text_area("Enter your review...")

        if st.button("Detect if review is unnatural..."):
            if review_text.strip() == '':
                st.error("Please enter a review")
            else:
                diff_days = datetime.datetime.today().date()  - review_date
                diff_days = diff_days.days
                test_df = combine_product_profile_df(current_product_df, current_profile_df)
                test_df, cleaned_text = custom_feature_engineering(test_df, review_rating, review_verified, review_votes, review_text,tfidf_normalizer)

                original_uninterested_columns = ['asin','acc_num','cleaned_reviews_profile_link','decoded_comment','cleaned_reviews_text','cleaned_reviews_date_posted','cleaned_reviews_location','locale','manual_label','fake_reviews','decision_function','impact_score','impact']
                original_trained_data = reviews_data.drop(columns=original_uninterested_columns)

                current_uninterested_columns = ['asin', 'description', 'price', 'rating', 'availability', 'decoded_comment', 'cleaned_products_text', 'name', 'url', 'json_data', 'acc_num', 'name', 'occupation', 'location', 'description', 'badges', 'ranking', 'reviewer_contributions', 'marketplace_id', 'locale', 'total_fake_reviews', 'proportion_fake_reviews', 'cleaned_profiles_not_brand_monogamist', 'cleaned_profiles_not_brand_loyalist', 'cleaned_profiles_not_brand_repeater', 'suspicious_reviewer_score']
                test_df = test_df.drop(columns=current_uninterested_columns)
                test_df = test_df[original_trained_data.columns]
                
                doc2vec_values = doc2vec_model.infer_vector(cleaned_text, steps=20, alpha=0.025)
                test_df = custom_merge_doc2vec(test_df, doc2vec_values)

                final_test_df = test_df.append(authentic_test_df)
                X_scaled = standard_normalizer.transform(final_test_df)
                X_normalized = normalize(X_scaled)

                with st.spinner("Predicting results...Please wait.."):
                    result, result_text, model_confidence = predict_custom_inputs(unnatural_reviewer_model, X_normalized, min(reviews_data['decision_function']), max(reviews_data['decision_function']))
                    impact_score, impact_text, impact_score_dict = evaluate_business_impact(max(reviews_data['cleaned_reviews_voting']), min(reviews_data['cleaned_reviews_voting']), review_votes, result, model_confidence, current_profile_df['suspicious_reviewer_score'].values[0], current_profile_df['proportion_fake_reviews'].values[0],diff_days)

                # st.success("Model Ran Successfully...")
                st.info("The review is **{}**. The model is ** {}% ** confident. The severity to the business is ** {} ({}%)**.".format(result_text,model_confidence*100, impact_text,round(impact_score,5)*100))
                with st.beta_expander("Breakdown of Model's Result (Top 10 Features)..."):
                    st.warning("The following breakdown is not the full list of features which was trained by the model...")
                    top10_features_mapping = {"cleaned_reviews_word_count":"Number of Words in Review", "cleaned_reviews_voting": "Number of Helpful Votes in Review", "cleaned_profiles_average_review_length": "Average Number of Words Written by Reviewer",
                                                "cleaned_profiles_average_helpfulVotes": "Average Number of Helpful Votes Received by Reviewer", "cleaned_reviews_sentiment": "Sentiment Score of Review",
                                                "cleaned_profiles_badges": "Number of Badges which the Reviewer has", "cleaned_profiles_proportion_same_day_reviews": "Proportion of Reviews Written by Reviewer on the Same Day",
                                                "cleaned_profiles_ranking": "Ranking of Reviewer on Amazon", "cleaned_products_total_reviews_posted": "Total Reviews found for Product", 
                                                "cleaned_products_total_never_verified_reviewer":"Number of Total Never-Verified Reviewers found for Product"}
                    breakdown_col1, breakdown_col2 = st.beta_columns(2)
                    odd_number_features = list(top10_features.keys())[::2]
                    even_number_features = list(top10_features.keys())[1::2]
                    with breakdown_col1:
                        start_index = 1
                        for index,column in enumerate(odd_number_features):
                            st.write("**{}) {}**".format(start_index, top10_features_mapping[column]))
                            data = list(authentic_test_df[column])
                            data.append(list(test_df[column])[0])
                            zscore = stats.zscore(data)
                            mean_value = round(np.mean(data),5)
                            first_text = "{} is {}.\n".format(top10_features_mapping[column], data[-1])
                            if zscore[-1] < 0:
                                second_text = "It is {} standard deviation **`below`** the mean value of {}.".format(abs(round(zscore[-1],5)),mean_value)
                            else:
                                second_text = "It is {} standard deviation **`above`** the mean value of {}.".format(abs(round(zscore[-1],5)),mean_value)
                            
                            abs_zscore = abs(zscore[-1])
                            if abs_zscore > 1 or abs_zscore < -1:
                                st.error("Failed...")
                            elif abs_zscore > 0.5 or abs_zscore <-0.5:
                                st.warning("Warning...")
                            else:
                                st.success("Pass...")
                            
                            st.write("This feature contributed **`{}%`** to the model's output.".format(round(top10_features[column]*100,5)))
                            st.write(first_text)
                            st.write(second_text)
                            st.write("")
                            start_index += 2

                    with breakdown_col2:
                        start_index = 2
                        for index,column in enumerate(even_number_features):
                            st.write("**{}) {}**".format(start_index, top10_features_mapping[column]))
                            data = list(authentic_test_df[column])
                            data.append(list(test_df[column])[0])
                            zscore = stats.zscore(data)
                            mean_value = round(np.mean(data),5)
                            first_text = "{} is {}.\n".format(top10_features_mapping[column], data[-1])
                            if zscore[-1] < 0:
                                second_text = "It is {} standard deviation **`below`** the mean value of {}.".format(abs(round(zscore[-1],5)),mean_value)
                            else:
                                second_text = "It is {} standard deviation **`above`** the mean value of {}.".format(abs(round(zscore[-1],5)),mean_value)
                            
                            abs_zscore = abs(zscore[-1])
                            if abs_zscore > 1 or abs_zscore < -1:
                                st.error("Failed...")
                            elif abs_zscore > 0.5 or abs_zscore <-0.5:
                                st.warning("Warning...")
                            else:
                                st.success("Pass...")
                            st.write("This feature contributed **`{}%`** to the model's output.".format(round(top10_features[column]*100,5)))
                            st.write(first_text)
                            st.write(second_text)
                            st.write("")
                            start_index += 2

                with st.beta_expander("Breakdown of Impact Scorer's Result..."):
                    review_centric_heuristics = list(impact_score_dict.keys())[:3]
                    profile_centric_heuristics = list(impact_score_dict.keys())[3:]
                    breakdown_col1, breakdown_col2 = st.beta_columns(2)
                    heuristic_explaination_mapping = {"Recency of Review": "The more recent the review is, the higher the score. If the review is less than or equals to 30 days old, it will have a score of 1. Else if the review is less than or equals to 90 days old, it will have a score of 0.8. Else if the review is less than or equals to 180 days old, it will have a score of 0.6. Else if the review is less than or equals to 360 days old, it will have a score of 0.4. Else if the review is less than or equals to 720 days old, it will have a score of 0.2. Else, it will have a score of 0.", 
                                        "Number of Helpful Votes": "The higher the helpful votes for the review, the higher the score.", 
                                        "Model’s Confidence of Review's Label": "A weighted score of the model’s confidence towards its output.", 
                                        "Suspicious Score of Reviewer": "Reviewers who are neither a brand repeater, loyalist or monogamists. They would be even more suspicious if they tend to post more than 1 reviews in a single day or has never written a verified purchaser review or has deleted reviews in their history.", 
                                        "Number of Unnatural Reviews Detected over Total Reviews Posted": "The higher the proportion of fake reviews posted by reviewer, the more suspicious the reviewer is."}
                    with breakdown_col1:
                        st.subheader("Review-Centric Heuristics")
                        st.write("")
                        for index, column in enumerate(review_centric_heuristics):
                            st.write("**{}) {}**".format(index+1,column))
                            st.markdown("**Description**: {}".format(heuristic_explaination_mapping[column]))
                            st.write("**Score**: {}".format(impact_score_dict[column]))
                            st.write('')
 
                    with breakdown_col2:
                        st.subheader("Profile-Centric Heuristics")
                        st.write("")
                        for index, column in enumerate(profile_centric_heuristics):
                            st.write("**{}) {}**".format(index+1,column))
                            st.markdown("**Description**: {}".format(heuristic_explaination_mapping[column]))
                            st.write("**Score**: {}".format(impact_score_dict[column]))
                            st.write('')
    
    for i in range(3): ## acting as a divider
        st.write('')

    with st.beta_container():
        st.header("📈 Interesting Insights")
        with st.beta_container():
            eda_col1, eda_col2 = st.beta_columns(2)
            with eda_col1:
                product_fake_reviews_df = reviews_data[ (reviews_data['asin'] == current_asin)].groupby('fake_reviews').size().reset_index(name='count')
                product_fake_reviews_df['fake_reviews'] = product_fake_reviews_df['fake_reviews'].replace({0:"Natural Reviews", 1: "Unnatural Reviews"})
                fig = px.pie(product_fake_reviews_df,values='count',names='fake_reviews', width=400, labels={'fake_reviews':"Type of Reviews",'count': "Number of Reviews"})
                st.header("Distribution of Reviews for this Product")
                st.plotly_chart(fig)  
                
            with eda_col2:
                profile_fake_reviews_df = reviews_data[ (reviews_data['acc_num'] == current_acc_num)].groupby('fake_reviews').size().reset_index(name='count')
                profile_fake_reviews_df['fake_reviews'] = profile_fake_reviews_df['fake_reviews'].replace({0:"Natural Reviews", 1: "Unnatural Reviews"})
                fig = px.pie(profile_fake_reviews_df,values='count',names='fake_reviews', width=400, labels={'fake_reviews':"Type of Reviews",'count': "Number of Reviews"})
                st.header("Distribution of Reviews for this Reviewer")
                st.plotly_chart(fig)  

        st.subheader("Product-Related")
        with st.beta_expander("Top 5 Natural Reviews for this Product"):
            current_reviews_df = reviews_data[ (reviews_data['asin'] == current_asin) & (reviews_data['fake_reviews'] == 0)].sort_values(by='decision_function', ascending=False)
            top_5_natural_results_df = current_reviews_df[:5]
            for index, row in top_5_natural_results_df.iterrows():
                top_5_natural_col1, top_5_natural_col2 = st.beta_columns(2)
                with top_5_natural_col1:
                    st.write("** Date Posted: ** {}".format(row['cleaned_reviews_date_posted']))
                    current_name = profile_inverse_mapping[row['acc_num']]
                    st.write("** Written by: ** {}".format(current_name))
                
                with top_5_natural_col2:
                    st.write("** Helpful Votes: ** {}".format(row['cleaned_reviews_voting']))
                    if row['cleaned_reviews_verified'] == 1:
                        st.write("** Verified Purchase: ** ✔️")
                    else:
                        st.write("** Verified Purchase: ** ❌")

                    st.write("** Model's Confidence: ** {}%".format(round(row['decision_function']*100,5)))
                
                with st.beta_container():
                    st.write("** Content: **")
                    st.write(row['decoded_comment'])

                st.write('')
            st.write('')

        with st.beta_expander("Top 5 Unnatural Reviews for this Product"):
            current_reviews_df = reviews_data[ (reviews_data['asin'] == current_asin) & (reviews_data['fake_reviews'] == 1)].sort_values(by='decision_function', ascending=False)
            top_5_unnatural_results_df = current_reviews_df[:5]
            for index, row in top_5_unnatural_results_df.iterrows():
                top_5_unnatural_col1, top_5_unnatural_col2 = st.beta_columns(2)
                with top_5_unnatural_col1:
                    st.write("** Date Posted: ** {}".format(row['cleaned_reviews_date_posted']))
                    current_name = profile_inverse_mapping[row['acc_num']]
                    st.write("** Written by: ** {}".format(current_name))
                
                with top_5_unnatural_col2:
                    st.write("** Helpful Votes: ** {}".format(row['cleaned_reviews_voting']))
                    if row['cleaned_reviews_verified'] == 1:
                        st.write("** Verified Purchase: ** ✔️")
                    else:
                        st.write("** Verified Purchase: ** ❌")
                    st.write("** Model's Confidence: ** {}%".format(round(row['decision_function']*100,5)))
                
                with st.beta_container():
                    st.write("** Content: **")
                    st.write(row['decoded_comment'])

                st.write('')
            st.write('')
        
        st.write('')
        st.write('')

        st.subheader("Reviewer-Related")
        with st.beta_expander("Top 5 Natural Reviews for this Reviewer"):
            current_reviews_df = reviews_data[ (reviews_data['acc_num'] == current_acc_num) & (reviews_data['fake_reviews'] == 0)].sort_values(by='decision_function', ascending=False)
            top_5_natural_results_df = current_reviews_df[:5]
            for index, row in top_5_natural_results_df.iterrows():
                top_5_natural_col1, top_5_natural_col2 = st.beta_columns(2)
                with top_5_natural_col1:
                    st.write("** Date Posted: ** {}".format(row['cleaned_reviews_date_posted']))
                    current_name = profile_inverse_mapping[row['acc_num']]
                    st.write("** Written by: ** {}".format(current_name))
                
                with top_5_natural_col2:
                    st.write("** Helpful Votes: ** {}".format(row['cleaned_reviews_voting']))
                    if row['cleaned_reviews_verified'] == 1:
                        st.write("** Verified Purchase: ** ✔️")
                    else:
                        st.write("** Verified Purchase: ** ❌")
                    st.write("** Model's Confidence: ** {}%".format(round(row['decision_function']*100,5)))
                
                with st.beta_container():
                    st.write("** Content: **")
                    st.write(row['decoded_comment'])

                st.write('')
            st.write('')

        with st.beta_expander("Top 5 Unnatural Reviews for this Reviewer"):
            current_reviews_df = reviews_data[ (reviews_data['acc_num'] == current_acc_num) & (reviews_data['fake_reviews'] == 1)].sort_values(by='decision_function', ascending=False)
            top_5_unnatural_results_df = current_reviews_df[:5]
            for index, row in top_5_unnatural_results_df.iterrows():
                top_5_unnatural_col1, top_5_unnatural_col2 = st.beta_columns(2)
                with top_5_unnatural_col1:
                    st.write("** Date Posted: ** {}".format(row['cleaned_reviews_date_posted']))
                    current_name = profile_inverse_mapping[row['acc_num']]
                    st.write("** Written by: ** {}".format(current_name))
                
                with top_5_unnatural_col2:
                    st.write("** Helpful Votes: ** {}".format(row['cleaned_reviews_voting']))
                    if row['cleaned_reviews_verified'] == 1:
                        st.write("** Verified Purchase: ** ✔️")
                    else:
                        st.write("** Verified Purchase: ** ❌")
                    st.write("** Model's Confidence: ** {}%".format(round(row['decision_function']*100,5)))
                
                with st.beta_container():
                    st.write("** Content: **")
                    st.write(row['decoded_comment'])

                st.write('')
            st.write('')

        
                


        


