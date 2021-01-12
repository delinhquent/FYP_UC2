import math

import numpy as np

import eif


class EIF:
    def __init__(self, model_config, model_df):
        self.model_config = model_config
        self.model_df = model_df
        self.model = None
    
    def make_eif(self):
        print("Performing Extended Isolation Forest...")
        user_ntrees = 100
        user_sample_size = 256
        self.model = eif.iForest(self.model_df.values, ntrees = user_ntrees, sample_size = user_sample_size, ExtensionLevel = self.model_df.shape[1] - 1)

        return {"ntree":user_ntrees, "sample_size":user_sample_size, "ExtensionLevel": self.model_df.shape[1] - 1}
    
    def predict_anomalies(self):
        expected_anomaly_ratio = self.model_config.eif.hyperparam_test.expected_anomaly_ratio
        
        # calculate anomaly scores
        anomaly_scores = self.model.compute_paths(X_in = self.model_df.values)

        # sort the scores
        anomaly_scores_sorted = np.argsort(anomaly_scores)

        indices_with_preds = anomaly_scores_sorted[-int(np.ceil(expected_anomaly_ratio * self.model_df.shape[0])):]

        # create predictions 
        y_pred = np.zeros_like([0]*len(self.model_df))
        y_pred[indices_with_preds] = 1

        return y_pred

    def evaluate_eif(self,results):
        print("Evaluating Isolation Forest...")
        total_reviews = len(list(results))
        total_fake_reviews = list(results).count(-1)
        total_non_fake_reviews = total_reviews - total_fake_reviews

        metrics = {"total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}
        
        return metrics
