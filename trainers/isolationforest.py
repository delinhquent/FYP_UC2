import eif

import math

import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from pyod.models.iforest import IForest

from utils.em_bench_high import calculate_emmv_score


class IsoForest:
    def __init__(self, model_config, model_df):
        self.model_config = model_config
        self.model_df = model_df
        self.model = None
        interested_columns = [column for column in self.model_df.columns if column != "manual_label"]
        self.modelling_data = self.model_df[interested_columns]
    
    def make_isolation_forest(self,model):
        if model == "eif":
            print("Performing Extended Isolation Forest...")
            user_ntrees = 100
            user_sample_size = 256
            self.model = eif.iForest(self.modelling_data.values, ntrees = user_ntrees, sample_size = user_sample_size, ExtensionLevel = self.modelling_data.shape[1] - 1)
            
            params = {"ntree":user_ntrees, "sample_size":user_sample_size, "ExtensionLevel": self.modelling_data.shape[1] - 1}
        elif model == "pyod_isolation_forest":
            self.model = IForest(random_state=self.model_config.isolation_forest.hyperparam_test.random_state).fit(self.modelling_data)
            params = self.model.get_params()
        else:
            print("Performing Isolation Forest...")
            self.model = IsolationForest(random_state=self.model_config.isolation_forest.hyperparam_test.random_state).fit(self.modelling_data)
            params = self.model.get_params()
        
        return params
    
    def predict_anomalies(self, model):
        if model == "eif":
            expected_anomaly_ratio = self.model_config.eif.hyperparam_test.expected_anomaly_ratio
        
            # calculate anomaly scores
            anomaly_scores = self.model.compute_paths(X_in = self.modelling_data.values)

            # sort the scores
            anomaly_scores_sorted = np.argsort(anomaly_scores)

            indices_with_preds = anomaly_scores_sorted[-int(np.ceil(expected_anomaly_ratio * self.modelling_data.shape[0])):]

            # create predictions 
            y_pred = np.zeros_like([0]*len(self.modelling_data))
            y_pred[indices_with_preds] = 1

            results = y_pred
            decisions = [None]*len(results)
            return results,decisions

        else:
            results = self.model.fit_predict(self.modelling_data)
            decisions = self.model.decision_function(self.modelling_data)

            return results, decisions

        

    def evaluate_isolation_forest(self,results, model):
        if model == "eif":
            print("Evaluating Extended Isolation Forest...")
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

            metrics = {"f1":f1Score,"precision":precScore,"recall":recallScore,"total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}
        else:
            total_reviews = len(list(results))
            if model == "isolation_forest":
                print("Evaluating Isolation Forest...")
                total_fake_reviews = list(results).count(-1)
            else:
                print("Evaluating PYOD Isolation Forest...")
                total_fake_reviews = list(results).count(1)
            total_non_fake_reviews = total_reviews - total_fake_reviews 

            if -1 in results:
                results = [1 if x == -1 else 0 for x in results]
            else:
                results = results

            self.model_df['results'] = list(results)
            evaluate_df = self.model_df[self.model_df['manual_label'].isin([0,1])]

            y_test = evaluate_df['manual_label']
            pred = evaluate_df['results']
            
            f1Score = f1_score(y_test, pred)
            recallScore = recall_score(y_test, pred)
            precScore = precision_score(y_test, pred) 
            
            em, mv = calculate_emmv_score(novelty_detection=False,ocsvm_model=False, X = self.modelling_data, y = results, model = self.model,model_name = model)

            metrics = {"f1":f1Score,"precision":precScore,"recall":recallScore,"em":em,"mv":mv, "total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}
        
        return metrics
        