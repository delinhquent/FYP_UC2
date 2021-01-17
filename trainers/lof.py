import math

import numpy as np

from trainers.lof_tuner import LOFAutoTuner

from sklearn.neighbors import LocalOutlierFactor

class LOF:
    def __init__(self, model_config, model_df):
        self.model_config = model_config
        self.model_df = model_df

    def hypertune_lof(self):
        print("Finding optimal k-neighbours and contaimination value for LOF Model...")
        tuner = LOFAutoTuner(data = self.model_df)

        params = tuner.run()

        return params

    def make_lof(self):
        params = self.hypertune_lof()
        self.model =  LocalOutlierFactor(n_neighbors=params['k'], contamination=params['c'])
        return self.model.get_params()
    
    def predict_anomalies(self):
        return self.model.fit_predict(self.model_df)

    def evaluate_lof(self, results):
        print("Evaluating LOF...")

        total_reviews = len(list(results))
        total_fake_reviews = list(results).count(-1)
        total_non_fake_reviews = total_reviews - total_fake_reviews

        metrics = {"total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}
        
        return metrics
