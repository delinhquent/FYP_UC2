from data_loader.data_loader import DataLoader

from featureselector.featureselector import FeatureSelector

import pandas as pd

import numpy as np

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize, StandardScaler 

from trainers.dbscan import DBScan
from trainers.isolationforest import IsoForest
from trainers.lof import LOF
from trainers.rrcf import RRCF
from trainers.pyodmodel import PyodModel


class Trainer:
    def __init__(self, **kwargs):
        valid_keys = ["config", "comet_config", "model_config", "experiment", "model", "text_represent", "feature_select","normalize"]
        for key in valid_keys:
            setattr(self, key, kwargs.get(key))
        self.model_data_loader = DataLoader(self.config.model.save_data_path)
        self.model_data = None
        self.tfidf_data = DataLoader(self.config.tfidf.reviews_vector).load_data()
        # self.glove_data = DataLoader(self.config.glove.reviews_vector).load_data()
        # self.fasttext_data = DataLoader(self.config.fasttext.reviews_vector).load_data()
        # self.word2vec_data = DataLoader(self.config.word2vec.reviews_vector).load_data()
        self.modelling_data = None
        self.trainer = None

    def load_data(self):
        self.model_data = self.get_model_data()

    def get_model_data(self):
        self.model_data_loader.load_data()
        return self.model_data_loader.get_data()
    
    def get_modelling_data(self):
        unnessary_columns = ['asin','acc_num','cleaned_reviews_profile_link','cleaned_reviews_text']
        modelling_df = self.model_data
        modelling_df = modelling_df.drop(columns=unnessary_columns)
        modelling_df = modelling_df.fillna(value=0)
        
        if self.feature_select == 'y':
            print("Proceeding with Feature Selection...")
            feature_selector = FeatureSelector(modelling_df)
            important_features = feature_selector.select_features()
            modelling_df = modelling_df[important_features]
        
        if self.normalize == 'y':
            print("Normalizing Data...")
            modelling_df = self.normalize_data()
        
        return self.combine_data(modelling_df)
                   

    def normalize_data(self):
        scaler = StandardScaler() 
        X_scaled = scaler.fit_transform(self.modelling_data) 

        # Normalizing the data so that the data, approximately follows a Gaussian distribution 
        X_normalized = normalize(X_scaled) 

        # Converting the numpy array into a pandas DataFrame 
        X_normalized = pd.DataFrame(X_normalized) 
        
        # Renaming the columns 
        X_normalized.columns = self.modelling_data.columns 
        
        return X_normalized

    def combine_data(self,modelling_df):
        print("Combining vectors with dataset...")
        if self.text_represent == 'tfidf':
            return pd.concat([modelling_df, self.tfidf_data],axis=1)
        elif self.text_represent == 'fasttext':
            return pd.concat([modelling_df, self.fasttext_data],axis=1)
        elif self.text_represent == 'glove':
            return pd.concat([modelling_df, self.glove_data],axis=1)
        elif self.text_represent == 'word2vec':
            return pd.concat([modelling_df, self.word2vec_data],axis=1)  

    def train_model(self):
        print("Retrieving necessary columns for modelling...")
        self.modelling_data = self.get_modelling_data()

        metrics, results = self.select_pipeline()
            
        self.model_data['fake_reviews'] = [1 if x == -1 else 0 for x in results]

        print("Saving results...")
        self.save_results(metrics)

    def select_pipeline(self):
        if self.model == "dbscan":
            metrics, results = self.dbscan_pipeline()
        elif self.model in ["isolation_forest","eif"]:
            metrics, results = self.isolation_forest_pipeline(self.model)
        elif self.model == "rrcf":
            metrics, results = self.rrcf_pipeline()
        elif self.model == "lof":
            metrics, results = self.lof_pipeline()
        elif self.model in ["ocsvm","copod", "hbos"]:
            metrics, results = self.generic_pyod_model_pipeline()
        return metrics, results

    def experiment_params(self,params):
        self.experiment.log_parameters(params)

    def save_results(self,metrics):
        self.experiment.log_metrics(metrics)
        
        results_path = {
            "dbscan" : self.model_config.dbscan.results.save_data_path,
            "isolation_forest": self.model_config.isolation_forest.results.save_data_path,
            "eif" : self.model_config.eif.results.save_data_path,
            "rrcf" : self.model_config.rrcf.results.save_data_path,
            "lof" : self.model_config.lof.results.save_data_path,
            "ocsvm" : self.model_config.ocsvm.results.save_data_path,
            "copod" : self.model_config.copod.results.save_data_path,
            "hbos" : self.model_config.hbos.results.save_data_path
            }

        self.model_data.to_csv(results_path[self.model],index=False)
        self.experiment.log_model(name=self.model,
                        file_or_folder=results_path[self.model])
        
    def dbscan_pipeline(self):
        print("Loading DBScan...")
        self.trainer = DBScan(model_config = self.model_config, model_df = self.modelling_data)
        params = self.trainer.hypertune_dbscan_params()

        print("Parsing parameters to Experiment...\nTesting parameters: {}".format(params))
        self.experiment_params(params)

        results = self.trainer.dbscan_cluster(params)
        
        self.model_data['dbscan_clusters'] = results

        metrics = self.trainer.evaluate_dbscan(results)
        
        return metrics, results
    
    def isolation_forest_pipeline(self, extended):
        if extended:
            print("Loading Extended Isolation Forest...")
        else:
            print("Loading Isolation Forest...")
        self.trainer = IsoForest(model_config = self.model_config, model_df = self.modelling_data)

        params = self.trainer.make_isolation_forest(extended)

        print("Parsing parameters to Experiment...\nTesting parameters: {}".format(params))
        self.experiment_params(params)
        
        results,decisions = self.trainer.predict_anomalies(extended)
        self.model_data[extended] = results
        self.model_data[extended+'_decision_function'] = decisions

        metrics = self.trainer.evaluate_isolation_forest(results,extended)
    
        return metrics, results

    def rrcf_pipeline(self):
        print("Loading Robust Random Cut Forest...")
        self.trainer = RRCF(model_config = self.model_config, model_df = self.modelling_data)

        params = self.trainer.make_rrcf()

        print("Parsing parameters to Experiment...\nTesting parameters: {}".format(params))
        self.experiment_params(params)

        codisp_results, results = self.trainer.predict_anomalies()

        self.model_data['rrcf'] = codisp_results

        metrics = self.trainer.evaluate_rrcf(results)
    
        return metrics, results
    
    def lof_pipeline(self):
        try:
            print("Loading Local Outlier Factor...")
            self.trainer = LOF(model_config = self.model_config, model_df = self.modelling_data)

            params = self.trainer.make_lof()

            print("Parsing parameters to Experiment...\nTesting parameters: {}".format(params))
            self.experiment_params(params)

            results,decisions = self.trainer.predict_anomalies()

            self.model_data['lof'] = results
            self.model_data['lof_decision_function'] = decisions

            metrics = self.trainer.evaluate_lof(results)
        
            return metrics, results
        except Exception as e:
            print(e)
    
    def generic_pyod_model_pipeline(self):
        name_dict = {"ocsvm":"One-Class SVM",
            "copod":"Copula Based Outlier Detector", "hbos": "Histogram-based Outlier Detection"}
        print("Loading {}...".format(name_dict[self.model]))
        self.trainer = PyodModel(model_config = self.model_config, model_df = self.modelling_data)
        params = self.trainer.make_pyod_model(self.model)

        print("Parsing parameters to Experiment...\nTesting parameters: {}".format(params))
        self.experiment_params(params)

        results,decisions = self.trainer.predict_anomalies()

        self.model_data[self.model] = results
        self.model_data[self.model+"_decision_function"] = decisions

        metrics = self.trainer.evaluate_pyod_model(results,name_dict[self.model])

        return metrics, results