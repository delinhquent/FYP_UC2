import math

import numpy as np

from trainers.lof_tuner import LOFAutoTuner

from joblib import dump, load

from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from pyod.models.lof import LOF

from utils.em_bench_high import calculate_emmv_score


class LocalOF:
    def __init__(self, model_config, model_df):
        self.model_config = model_config
        self.model_df = model_df
        self.model = None
        interested_columns = [column for column in self.model_df.columns if column != "manual_label"]
        self.modelling_data = self.model_df[interested_columns]

    def hypertune_lof(self,model):
        print("Finding optimal k-neighbours and contaimination value for LOF Model...")
        tuner = LOFAutoTuner(data = self.modelling_data)

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
        self.model = self.model.fit(self.modelling_data)
        results = self.model.predict(self.modelling_data)
        decisions = self.model.decision_function(self.modelling_data)
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

        em, mv = calculate_emmv_score(novelty_detection=False,ocsvm_model=False, X = self.modelling_data, y = results, model = self.model, model_name = model)

        metrics = {"f1":f1Score,"precision":precScore,"recall":recallScore,"em":em,"mv":mv,"total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}
        
        print("Saving model...")
        dump(self.model, "models/" + str(model) + ".joblib")

        return metrics
