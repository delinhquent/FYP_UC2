import math

import numpy as np

from pyod.models.ocsvm import OCSVM

from trainers.ocsvm_tuner import find_best_ocsvm
from utils.em_bench_high import calculate_emmv_score


class OneClassSVM:
    def __init__(self, model_config, model_df):
        self.model_config = model_config
        self.model_df = model_df

    def hypertune_ocsvm(self, results):
        print("Finding optimal nu and gamma value for One-Class Support Vector Machine...")
        
        self.model = find_best_ocsvm(self.model_df.values, results)
        
        return self.model.get_params()
    
    def make_ocsvm(self):
        # create a default ocsvm to generate y labels for hypertuning
        self.model = OCSVM()
    
    def predict_anomalies(self):
        self.model.fit(self.model_df)
        results = self.model.labels_
        decisions = self.model.decision_function(self.model_df)
        
        return results, decisions

    def evaluate_ocsvm(self, results):
        print("Evaluating One-Class Support Vector Machine...")

        total_reviews = len(list(results))
        total_fake_reviews = list(results).count(1)
        total_non_fake_reviews = total_reviews - total_fake_reviews
        
        em, mv = calculate_emmv_score(novelty_detection=False,ocsvm_model=True, X = self.model_df, y = results, model = self.model)

        metrics = {"em":em,"mv":mv,"total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}

        return metrics
