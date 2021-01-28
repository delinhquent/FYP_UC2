import eif

import math

import numpy as np

from sklearn.ensemble import IsolationForest

from utils.em_bench_high import calculate_emmv_score


class IsoForest:
    def __init__(self, model_config, model_df):
        self.model_config = model_config
        self.model_df = model_df
        self.model = None
    
    def make_isolation_forest(self,extended):
        if extended == "eif":
            print("Performing Extended Isolation Forest...")
            user_ntrees = 100
            user_sample_size = 256
            self.model = eif.iForest(self.model_df.values, ntrees = user_ntrees, sample_size = user_sample_size, ExtensionLevel = self.model_df.shape[1] - 1)
            
            params = {"ntree":user_ntrees, "sample_size":user_sample_size, "ExtensionLevel": self.model_df.shape[1] - 1}
        else:
            print("Performing Isolation Forest...")
            self.model = IsolationForest(random_state=self.model_config.isolation_forest.hyperparam_test.random_state).fit(self.model_df)
            params = self.model.get_params()
        
        return params
    
    def predict_anomalies(self, extended):
        if extended == "eif":
            expected_anomaly_ratio = self.model_config.eif.hyperparam_test.expected_anomaly_ratio
        
            # calculate anomaly scores
            anomaly_scores = self.model.compute_paths(X_in = self.model_df.values)

            # sort the scores
            anomaly_scores_sorted = np.argsort(anomaly_scores)

            indices_with_preds = anomaly_scores_sorted[-int(np.ceil(expected_anomaly_ratio * self.model_df.shape[0])):]

            # create predictions 
            y_pred = np.zeros_like([0]*len(self.model_df))
            y_pred[indices_with_preds] = 1

            results = y_pred
            decisions = [None]*len(results)
            return results,decisions

        else:
            results = self.model.fit_predict(self.model_df)
            decisions = self.model.decision_function(self.model_df)

            return results, decisions

        

    def evaluate_isolation_forest(self,results, extended):
        if extended == "eif":
            print("Evaluating Extended Isolation Forest...")
            total_reviews = len(list(results))
            total_fake_reviews = list(results).count(-1)
            total_non_fake_reviews = total_reviews - total_fake_reviews

            metrics = {"total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}
        else:
            print("Evaluating Isolation Forest...")
            total_reviews = len(list(results))
            total_fake_reviews = list(results).count(-1)
            total_non_fake_reviews = total_reviews - total_fake_reviews

            results_mean = np.mean(results)
            results_var = np.var(results)

            em, mv = calculate_emmv_score(self.model_df, results, self.model, novelty_detection=False,ocsvm_model=False)

            metrics = {"em":em,"mv":mv,"results_mean": results_mean, "results_var":results_var, "total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}
        
        return metrics
        