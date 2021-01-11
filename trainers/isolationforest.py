import math

import numpy as np

from sklearn.ensemble import IsolationForest


class IsoForest:
    def __init__(self, model_config, model_df):
        self.model_config = model_config
        self.model_df = model_df
        self.model = None
    
    def make_isolation_forest(self):
        print("Performing Isolation Forest...")
        self.model = IsolationForest(random_state=self.model_config.isolation_forest.hyperparam_test.random_state).fit(self.model_df)
        return self.model.get_params()
    
    def predict_anomalies(self):
        return self.model.predict(self.model_df)

    def evaluate_isolation_forest(self,results):
        print("Evaluating Isolation Forest...")
        total_reviews = len(list(results))
        total_fake_reviews = list(results).count(-1)
        total_non_fake_reviews = total_reviews - total_fake_reviews

        results_mean = np.mean(results)
        results_var = np.var(results)

        metrics = {"results_mean": results_mean, "results_var":results_var, "total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}
        
        return metrics
