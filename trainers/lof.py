import math

import numpy as np

from trainers.lof_tuner import LOFAutoTuner

from joblib import dump, load

from sklearn.neighbors import LocalOutlierFactor
from pyod.models.lof import LOF

from utils.em_bench_high import calculate_emmv_score


class LocalOF:
    def __init__(self, model_config, model_df):
        self.model_config = model_config
        self.model_df = model_df
        self.model = None

    def hypertune_lof(self,model):
        print("Finding optimal k-neighbours and contaimination value for LOF Model...")
        tuner = LOFAutoTuner(data = self.model_df)

        params = tuner.run()

        return params

    def make_lof(self,model):
        params = self.hypertune_lof(model)
        if model == "lof":
            self.model =  LocalOutlierFactor(n_neighbors=params['k'], contamination=params['c'],novelty=True)
        else:
            self.model = LOF(n_neighbors=params['k'], contamination=params['c'])
        return self.model.get_params()
    
    def predict_anomalies(self):
        self.model = self.model.fit(self.model_df)
        results = self.model.predict(self.model_df)
        decisions = self.model.decision_function(self.model_df)
        return results, decisions

    def evaluate_lof(self, results,model):

        total_reviews = len(list(results))
        if model == "lof":
            print("Evaluating LOF...")
            total_fake_reviews = list(results).count(-1)
        else:
            print("Evaluating PYOD LOF...")
            total_fake_reviews = list(results).count(1)

        total_non_fake_reviews = total_reviews - total_fake_reviews

        em, mv = calculate_emmv_score(novelty_detection=False,ocsvm_model=False, X = self.model_df, y = results, model = self.model, model_name = model)

        metrics = {"em":em,"mv":mv,"total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}
        
        print("Saving model...")
        dump(self.model.fit(self.model_df), "models/" + str(model) + ".joblib")

        return metrics
