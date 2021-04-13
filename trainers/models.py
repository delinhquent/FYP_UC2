import math

import numpy as np

from pyod.models.hbos import HBOS
from pyod.models.copod import COPOD
from pyod.models.ocsvm import OCSVM
from sklearn.ensemble import IsolationForest
from pyod.models.iforest import IForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.lof import LOF

from utils.em_bench_high import calculate_emmv_score
from trainers.ocsvm_tuner import *

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier

import pandas as pd

from joblib import dump, load
import pickle

from imblearn.over_sampling import SMOTE
import operator
import json


class Model:
    def __init__(self, model_config, model_df):
        self.model_config = model_config
        self.model_df = model_df
        self.original_train_df, self.original_test_df, self.train_df, self.test_df, self.train_index, self.test_index = self.generate_train_test(model_df)
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

        unnessary_columns = ['index','asin','acc_num','cleaned_reviews_profile_link','decoded_comment','cleaned_reviews_text','cleaned_reviews_date_posted','cleaned_reviews_location','locale']
    
        train_df = original_train_df.copy()
        train_index = train_df['index']
        train_df = train_df.drop(columns=unnessary_columns).fillna(0)

        test_df = original_test_df.copy()
        test_index = test_df['index']
        test_df = test_df.drop(columns=unnessary_columns).fillna(0)

        return original_train_df, original_test_df, train_df, test_df, train_index, test_index

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

    def make_model(self,model):
        if model == "hbos":
            self.model =  HBOS()
        elif model == "copod":
            self.model = COPOD()
        elif model == "ocsvm":
            self.model = OCSVM()
        elif model == "pyod_isolation_forest":
            self.model = IForest()
        elif model == "isolation_forest":
            self.model = IsolationForest()
        elif model == "lof":
            self.model =  LocalOutlierFactor(novelty=True)
        elif model == "pyod_lof":
            self.model = LOF()
        return self.model.get_params()
    
    def train_model(self):
        print("Identifying outliers...")
        y_train = self.train_df['manual_label']
        X_train = self.train_df.drop(columns='manual_label')

        # self.model.fit(X_train)
        sample_size = 10000 

        sm = SMOTE(sampling_strategy={0: sample_size,1: sample_size}, random_state=0)
        augmented_y_train= y_train
        augmented_y_train[:len(X_train)//2] = 1
        X_train_res, y_train_res = sm.fit_sample(X_train, augmented_y_train)
        print("Fitting Train Dataset with Dataset Size {}...".format(X_train_res.shape))
        self.model.fit(X_train_res)
        print("Saving model...")
        pickle.dump(self.model, open('models/ocsvm.pkl','wb'))
        self.original_train_df['fake_reviews'] = y_train
        self.original_train_df['index'] = self.train_index

    def evaluate_model(self, model_name, model):
        print("Evaluating {}...".format(model_name))
        X_test = self.test_df.drop(columns='manual_label')
        
        print("Predicting Test Dataset with Dataset Size {}...".format(X_test.shape))
        results = self.model.predict(X_test)
        if -1 in results:
            results = [1 if x == -1 else 0 for x in results]
        else:
            results = results

        self.original_test_df['fake_reviews'] = results
        self.original_test_df['decision_function'] = self.model.decision_function(X_test)
        self.original_test_df['index'] = self.test_index
        self.original_train_df['decision_function'] = max(self.original_test_df['decision_function'])
        self.train_df['fake_reviews'] = 0
        self.test_df['fake_reviews'] = results

        evaluate_df = self.original_test_df[self.original_test_df['manual_label'].notnull()]
        y_test = evaluate_df['manual_label']
        pred = evaluate_df['fake_reviews']

        f1Score = f1_score(y_test, pred)
        recallScore = recall_score(y_test, pred)
        precScore = precision_score(y_test, pred)
        
        metrics = {"f1":f1Score,"precision":precScore,"recall":recallScore}

        final_df = self.original_train_df.append(self.original_test_df, ignore_index=True)
        final_df = final_df.sort_values(by=['index'])
        final_df = final_df.drop(columns='index')
        final_df['decision_function'] = final_df['decision_function'].fillna(max(self.original_test_df['decision_function']))
        final_df['fake_reviews'] = final_df['fake_reviews'].fillna(0)

        print("Conducting feature importance...")
        rf_df = self.train_df.append(self.test_df, ignore_index=True)
        rf = ExtraTreesClassifier(n_estimators=250,
                            random_state=0)
        y = rf_df['fake_reviews']
        X = rf_df.drop(columns=['fake_reviews','manual_label'])
        rf.fit(X, y)
        pickle.dump(rf, open('models/rf.pkl','wb'))

        importances = rf.feature_importances_

        importance_dict = dict(zip(X.columns, importances))
        new_importance_dict = dict( sorted(importance_dict.items(), key=operator.itemgetter(1),reverse=True)[:10])
        filename = "models/results/{}_feature_importance.json".format(model)
        with open(filename, 'w') as fp:
            json.dump(new_importance_dict, fp)
        
        return metrics, final_df.fillna(0)