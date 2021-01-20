import math

import numpy as np

from pyod.models.hbos import HBOS
from pyod.models.copod import COPOD
from pyod.models.ocsvm import OCSVM


class PyodModel:
    def __init__(self, model_config, model_df):
        self.model_config = model_config
        self.model_df = model_df

    def make_pyod_model(self,model):
        if model == "hbos":
            self.model =  HBOS()
        elif model == "copod":
            self.model = COPOD()
        elif model == "ocsvm":
            self.model = OCSVM()
        return self.model.get_params()
    
    def predict_anomalies(self):
        self.model.fit(self.model_df)
        return self.model.labels_

    def evaluate_pyod_model(self, results, model_name):
        print("Evaluating {}...".format(model_name))

        total_reviews = len(list(results))
        total_fake_reviews = list(results).count(-1)
        total_non_fake_reviews = total_reviews - total_fake_reviews

        metrics = {"total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}
        
        return metrics