import math

import numpy as np

from pyod.models.hbos import HBOS
from pyod.models.copod import COPOD
from pyod.models.ocsvm import OCSVM

from utils.em_bench_high import calculate_emmv_score
from trainers.ocsvm_tuner import find_best_ocsvm

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

from joblib import dump, load


class PyodModel:
    def __init__(self, model_config, model_df):
        self.model_config = model_config
        self.model_df = model_df
        self.model = None
        interested_columns = [column for column in self.model_df.columns if column != "manual_label"]
        self.modelling_data = self.model_df[interested_columns]

    def hypertune_ocsvm(self, results):
        print("Finding optimal nu and gamma value for One-Class Support Vector Machine...")
        
        evaluate_df = self.model_df[self.model_df['manual_label'].isin([0,1])]
        X = evaluate_df.values
        y = evalaute_df['manual_label'].values

        # self.model = find_best_ocsvm(self.model_df.values, results)
        self.model = find_best_ocsvm(X, y)
        
        return self.model.get_params()
    
    def make_pyod_model(self,model):
        if model == "hbos":
            self.model =  HBOS()
        elif model == "copod":
            self.model = COPOD()
        elif model == "ocsvm":
            self.model = OCSVM()
        return self.model.get_params()
    
    def predict_anomalies(self):
        print("Identifying outliers...")
        self.model.fit(self.modelling_data)
        results = self.model.labels_
        decisions = self.model.decision_function(self.modelling_data)
        
        return results, decisions

    def evaluate_pyod_model(self, results, model_name, model):
        print("Evaluating {}...".format(model_name))

        total_reviews = len(list(results))
        total_fake_reviews = list(results).count(1)
        total_non_fake_reviews = total_reviews - total_fake_reviews

        if -1 in results:
            results = [1 if x == -1 else 0 for x in results]
        else:
            results = results

        self.model_df['results'] = results
        evaluate_df = self.model_df[self.model_df['manual_label'].isin([0,1])]

        y_test = evaluate_df['manual_label']
        pred = evaluate_df['results']

        f1Score = f1_score(y_test, pred)
        recallScore = recall_score(y_test, pred)
        precScore = precision_score(y_test, pred)  

        if model_name == "One-Class SVM":
            em, mv = calculate_emmv_score(novelty_detection=False, ocsvm_model=True, X = self.modelling_data, y = results, model = self.model, model_name = model)
            print("Saving model...")
            dump(self.model, "models/" + str(model) + ".joblib")
        
        else:
            em, mv = calculate_emmv_score(novelty_detection=False, ocsvm_model=False, X = self.modelling_data, y = results, model = self.model, model_name = model)

        metrics = {"f1":f1Score,"precision":precScore,"recall":recallScore,"em":em,"mv":mv,"total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}

        return metrics
