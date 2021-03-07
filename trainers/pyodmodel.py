import math

import numpy as np

from pyod.models.hbos import HBOS
from pyod.models.copod import COPOD
from pyod.models.ocsvm import OCSVM

from utils.em_bench_high import calculate_emmv_score
from trainers.ocsvm_tuner import *

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

import pandas as pd

from joblib import dump, load

from imblearn.over_sampling import SMOTE


class PyodModel:
    def __init__(self, model_config, model_df):
        self.model_config = model_config
        self.original_train_df, self.original_test_df, self.train_df, self.test_df = self.generate_train_test(model_df)
        self.model = None

    def generate_train_test(self, model_df):
        original_train_df = model_df[model_df['manual_label'] == 0]
        original_test_df = model_df[~(model_df['manual_label'] == 0)]

        y = original_train_df['manual_label']
        X = original_train_df.drop(columns='manual_label')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        original_train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)

        temp_test_df = pd.merge(X_test, y_test, left_index=True, right_index=True)
        original_test_df = original_test_df.append(temp_test_df, ignore_index=True)

        unnessary_columns = ['asin','acc_num','cleaned_reviews_profile_link','decoded_comment','cleaned_reviews_text','cleaned_reviews_date_posted','locale']

        train_df = original_train_df.copy()
        train_df = train_df.drop(columns=unnessary_columns).fillna(0)

        
        test_df = original_test_df.copy()
        test_df = test_df.drop(columns=unnessary_columns).fillna(0)

        return original_train_df, original_test_df, train_df, test_df

    def hypertune_ocsvm(self):
    # def hypertune_ocsvm(self):
        print("Finding optimal kernel, nu and gamma value for One-Class Support Vector Machine...")
        X_train = self.train_df.drop(columns='manual_label')
        
        # evaluate_df = self.original_test_df[self.original_test_df['manual_label'].notnull()]
        # y_test = evaluate_df['manual_label']
        # X_test = evaluate_df.drop(columns='manual_label')

        self.model = find_best_ocsvm(X_train).fit(X_train)
        # self.model = find_best_ocsvm_adapted(X_train,evaluate_df).fit(X_train)
        self.original_train_df['fake_reviews'] = self.model.labels_
        self.original_train_df['decision_function'] = self.model.decision_function(X_train)
        return self.model

    def make_pyod_model(self,model):
        if model == "hbos":
            self.model =  HBOS()
        elif model == "copod":
            self.model = COPOD()
        elif model == "ocsvm":
            self.model = OCSVM()
            # self.model = self.hypertune_ocsvm()
        return self.model.get_params()
    
    def predict_anomalies(self):
        print("Identifying outliers...")
        y_train = self.train_df['manual_label']
        X_train = self.train_df.drop(columns='manual_label')

        # self.model.fit(X_train)
        sample_size = 50000 

        sm = SMOTE(sampling_strategy={0: sample_size,1: sample_size}, random_state=0)
        augmented_y_train= y_train
        augmented_y_train[:len(X_train)//2] = 1
        X_train_res, y_train_res = sm.fit_sample(X_train, augmented_y_train)
        print("Fitting Train Dataset with Dataset Size {}...".format(X_train_res.shape))
        self.model.fit(X_train_res)
        self.original_train_df['fake_reviews'] = y_train
        self.original_train_df['decision_function'] = max(self.model.decision_function(X_train_res))

    def evaluate_pyod_model(self, model_name, model):
        print("Evaluating {}...".format(model_name))
        X_test = self.test_df.drop(columns='manual_label')
        
        print("Predicting Test Dataset with Dataset Size {}...".format(X_test.shape))
        results = self.model.predict(X_test)
        
        self.original_test_df['fake_reviews'] = results
        self.original_test_df['decision_function'] = self.model.decision_function(X_test)

        evaluate_df = self.original_test_df[self.original_test_df['manual_label'].notnull()]
        y_test = evaluate_df['manual_label']
        pred = evaluate_df['fake_reviews']

        f1Score = f1_score(y_test, pred)
        recallScore = recall_score(y_test, pred)
        precScore = precision_score(y_test, pred)

        if model_name == "One-Class SVM":
            em, mv = calculate_emmv_score(novelty_detection=False, ocsvm_model=True, X = X_test, y = results, model = self.model, model_name = model)
            print("Saving model...")
            dump(self.model, "models/" + str(model) + ".joblib")
        
        else:
            em, mv = calculate_emmv_score(novelty_detection=False, ocsvm_model=False, X = X_test, y = results, model = self.model, model_name = model)

        metrics = {"f1":f1Score,"precision":precScore,"recall":recallScore,"em":em,"mv":mv}

        final_df = self.original_train_df.append(self.original_test_df, ignore_index=True)

        return metrics, final_df
