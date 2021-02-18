import math

import numpy as np

import pandas as pd

from sklearn.feature_selection import VarianceThreshold

import json


class FeatureSelector:
    def __init__(self, df):
        self.df = df

    def select_features(self):
        total_features = len(self.df.columns)
        print("Total Features to proceed with Feature Selection: {}...".format(total_features))

        # print("Selecting Important Features using Variance Threshold...")
        # current_features = self.variance()
        # print("Proceeding with Feature Selection using Dispersion Ratio...")
        # current_features = self.dispersion_ratio()
        f = open('models/results/baseline_ocsvm_feature_importance.json',)
        d = json.load(f)
        # current_features = list(d.keys())
        max_value = max(d.values())
        d = {k: v / max_value for k, v in d.items()}
        current_features = [k for k, v in d.items() if v >= 0.25]
        return current_features

    def variance(self):
        v_threshold = VarianceThreshold(threshold=0)
        v_threshold.fit(self.df)
        importance_by_variance = v_threshold.get_support()
        importance_by_variance_dict = dict(zip(self.df.columns,importance_by_variance))
        important_features = list(importance_by_variance).count(True)
        print("Number of important features: {}\n".format(important_features))

        current_features = []
        for key,value in importance_by_variance_dict.items():
            if value == True:
                current_features.append(key)
        
        return current_features
    
    def dispersion_ratio(self):
        df_values = self.df.values + 1
        aritmeticMean = np.mean(df_values, axis =0 )
        geometricMean = np.power(np.prod(df_values, axis =0 ),1/df_values.shape[0])
        R = aritmeticMean/geometricMean
        
        current_features = []
        for i in range(len(R)):
            dispersion_value = R[i]
            if dispersion_value >= 1:
                current_features.append(self.df.columns[i])
        
        print("Number of important features by dispersion ratio: {}".format(len(current_features)))

        return current_features
        
    
