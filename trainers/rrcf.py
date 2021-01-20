import numpy as np

import rrcf


class RRCF:
    def __init__(self, model_config, model_df):
        self.model_config = model_config
        self.model_df = model_df
        self.average_codisp_results = None

    def make_rrcf(self):
        params = {}
        self.model = rrcf.RCTree(self.model_df)
        return params
    
    def predict_anomalies(self):
        codisp_results = []
        for index in range(len(self.model_df)):
            codisp_results.append(self.model.codisp(index))
        
        mean = np.mean(codisp_results, axis=0)
        sd = np.std(codisp_results, axis=0)

        self.average_codisp_results = mean

        results = [1 if (x > mean - 3 * sd) else 0 for x in codisp_results ]
        results = [1 if (x < mean + 3 * sd) else 0 for x in results]

        return codisp_results, results

    def evaluate_rrcf(self, results):
        print("Evaluating Robust Random Cut Forest...")

        total_reviews = len(list(results))
        total_fake_reviews = list(results).count(-1)
        total_non_fake_reviews = total_reviews - total_fake_reviews

        metrics = {"average_codisp_results":self.average_codisp_results,"total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}
        
        return metrics
        