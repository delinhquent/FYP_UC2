from data_loader.data_loader import DataLoader

from featureselector.featureselector import FeatureSelector

import pandas as pd
import pickle


import numpy as np

from impactscorer.impactscorer import ImpactScorer

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize, StandardScaler

from trainers.dbscan import DBScan
from trainers.isolationforest import IsoForest
from trainers.lof import LocalOF
from trainers.rrcf import RRCF
from trainers.pyodmodel import PyodModel

from gensim.models.doc2vec import Doc2Vec

class Trainer:
    def __init__(self, **kwargs):
        valid_keys = ["config", "comet_config", "model_config", "experiment", "model", "text_represent", "feature_select","normalize"]
        for key in valid_keys:
            setattr(self, key, kwargs.get(key))
        self.model_data_loader = DataLoader(self.config.model.save_data_path)
        self.model_data = None
        self.profile_data_loader = DataLoader(self.config.profiles.interim_data_path)
        self.profile_data = None
        self.doc2vec_data_loader = DataLoader(self.config.doc2vec.reviews_vector)
        self.doc2vec_data = None
        self.modelling_data = None
        self.trainer = None

    def load_data(self):
        self.model_data = self.get_model_data()
        self.doc2vec_data = self.get_dov2vec_data()
        self.profile_data = self.get_profile_data()

    def get_model_data(self):
        self.model_data_loader.load_data()
        return self.model_data_loader.get_data()

    def get_tfidf_data(self):
        self.tfidf_data_loader.load_data()
        return self.tfidf_data_loader.get_data()

    def get_dov2vec_data(self):
        self.doc2vec_data_loader.load_data()
        return self.doc2vec_data_loader.get_data()

    def get_profile_data(self):
        self.profile_data_loader.load_data()
        return self.profile_data_loader.get_data()
    
    def get_modelling_data(self):
        self.modelling_data = self.model_data
        
        print("Combining vectors with dataset...")
        if self.text_represent == 'tfidf':
            self.modelling_data = pd.merge(self.modelling_data, self.tfidf_data, left_index=True, right_index=True)
        elif self.text_represent == 'fasttext':
            self.modelling_data = pd.merge(self.modelling_data, self.fasttext_data, left_index=True, right_index=True)
        elif self.text_represent == 'glove':
            self.modelling_data = pd.merge(self.modelling_data, self.glove_data, left_index=True, right_index=True)
        elif self.text_represent == 'word2vec':
            self.modelling_data = pd.merge(self.modelling_data, self.word2vec_data, left_index=True, right_index=True)
        elif self.text_represent == 'doc2vec':
            self.modelling_data = pd.merge(self.modelling_data, self.doc2vec_data, left_index=True, right_index=True)

        self.modelling_data = self.modelling_data[self.modelling_data['asin'].notnull()]
        self.modelling_data = self.modelling_data[self.modelling_data['acc_num'].notnull()]
        self.modelling_data = self.modelling_data[self.modelling_data['cleaned_reviews_text'].notnull()]
        self.modelling_data['index'] = self.modelling_data.index
        print("Current dataset size after dropping null values: {}".format(self.modelling_data.shape))

        unnessary_columns = ['asin','acc_num','cleaned_reviews_profile_link','decoded_comment','cleaned_reviews_text','cleaned_reviews_date_posted','locale','manual_label']
        unnecessary_df = self.modelling_data[['index'] + unnessary_columns]

        modelling_df = self.modelling_data
        modelling_df = modelling_df.drop(columns=unnessary_columns)
        modelling_df = modelling_df.fillna(0)
        print("Dataset for feature selection and (or) normalization: {}".format(modelling_df.shape))


        if self.feature_select == 'y':
            print("Proceeding with Feature Selection...")
            feature_selector = FeatureSelector(modelling_df)
            important_features = feature_selector.select_features()

            modelling_df = modelling_df[important_features]
        
        if self.normalize == 'y':
            print("Normalizing Data...")
            index = list(modelling_df['index'])
            modelling_df = self.normalize_data(modelling_df.drop(columns='index'))
            modelling_df['index'] = index
        
        self.modelling_data = pd.merge(unnecessary_df,modelling_df,left_on=['index'], right_on = ['index'], how = 'left')
        self.modelling_data = self.modelling_data.drop(columns='index')
        print(self.modelling_data.shape)


    def normalize_data(self,modelling_df):
        scaler = StandardScaler() 
        X_scaled = scaler.fit_transform(modelling_df) 
        pickle.dump(X_scaled, open('models/normalizer/feature_normalizer_standard.pkl','wb'))
        # sc = pickle.load(open('file/path/scaler.pkl','rb')) # keeping this here for future development
        
        # Normalizing the data so that the data, approximately follows a Gaussian distribution 
        X_normalized = normalize(X_scaled) 

        # Converting the numpy array into a pandas DataFrame 
        X_normalized = pd.DataFrame(X_normalized) 
        
        # Renaming the columns 
        X_normalized.columns = modelling_df.columns 

        return X_normalized

    def train_model(self):
        print("Retrieving necessary columns for modelling...")
        self.get_modelling_data()

        metrics, self.model_data = self.select_pipeline()

        print("Assessing impact...")
        assessor = ImpactScorer(self.model_data,self.profile_data)

        self.model_data, self.profile_data = assessor.assess_impact()

        print("Saving results...")
        self.save_results(metrics)
        interested_columns = ['acc_num','proportion_fake_reviews','suspicious_reviewer_score']
        self.profile_data[interested_columns].to_csv(self.config.profiles.save_data_path,index=False)

        print("Training of Model Completed...")

    def select_pipeline(self):
        if self.model == "dbscan":
            metrics, results = self.dbscan_pipeline()
        elif self.model in ["isolation_forest","eif","pyod_isolation_forest"]:
            metrics, results = self.isolation_forest_pipeline(self.model)
        elif self.model == "rrcf":
            metrics, results = self.rrcf_pipeline()
        elif self.model in ["lof","pyod_lof"]:
            metrics, results = self.lof_pipeline(self.model)
        elif self.model in ["copod", "hbos"]:
            metrics, results = self.generic_pyod_model_pipeline()
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
            "hbos" : self.model_config.hbos.results.save_data_path,
            "pyod_isolation_forest" : self.model_config.pyod_isolation_forest.results.save_data_path,
            "pyod_lof" : self.model_config.pyod_lof.results.save_data_path
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
        
        self.model_data['model_output'] = results

        metrics = self.trainer.evaluate_dbscan(results)
        
        return metrics, results
    
    def isolation_forest_pipeline(self, model):
        if model == "eif":
            print("Loading Extended Isolation Forest...")
        elif model == "pyod_isolation_forest":
            print("Loading PYOD Isolation Forest...")
        else:
            print("Loading Isolation Forest...")
        self.trainer = IsoForest(model_config = self.model_config, model_df = self.modelling_data)

        params = self.trainer.make_isolation_forest(model)

        print("Parsing parameters to Experiment...\nTesting parameters: {}".format(params))
        self.experiment_params(params)
        
        results,decisions = self.trainer.predict_anomalies(model)
        
        self.model_data["model_output"] = results
        self.model_data['decision_function'] = decisions

        metrics = self.trainer.evaluate_isolation_forest(results,model)
    
        return metrics, results

    def rrcf_pipeline(self):
        print("Loading Robust Random Cut Forest...")
        self.trainer = RRCF(model_config = self.model_config, model_df = self.modelling_data)

        params = self.trainer.make_rrcf()

        print("Parsing parameters to Experiment...\nTesting parameters: {}".format(params))
        self.experiment_params(params)

        codisp_results, results = self.trainer.predict_anomalies()

        self.model_data['model_output'] = codisp_results

        metrics = self.trainer.evaluate_rrcf(results)
    
        return metrics, results
    
    def lof_pipeline(self,model):
        if model == "lof":
            print("Loading Local Outlier Factor...")
        else:
            print("Loading PYOD Local Outlier Factor...")
        self.trainer = LocalOF(model_config = self.model_config, model_df = self.modelling_data)

        params = self.trainer.make_lof(model)

        print("Parsing parameters to Experiment...\nTesting parameters: {}".format(params))
        self.experiment_params(params)

        results,decisions = self.trainer.predict_anomalies()
        
        self.model_data['model_output'] = results
        self.model_data['decision_function'] = decisions


        metrics = self.trainer.evaluate_lof(results,model)
        return metrics, results
    
    def generic_pyod_model_pipeline(self):
        name_dict = {"ocsvm":"One-Class SVM",
            "copod":"Copula Based Outlier Detector", "hbos": "Histogram-based Outlier Detection"}
        print("Loading {}...".format(name_dict[self.model]))
        self.trainer = PyodModel(model_config = self.model_config, model_df = self.modelling_data)

        params = self.trainer.make_pyod_model(self.model)

        print("Parsing parameters to Experiment...\nTesting parameters: {}".format(params))
        self.experiment_params(params)

        self.trainer.predict_anomalies()

        metrics, self.model_data = self.trainer.evaluate_pyod_model(name_dict[self.model],self.model)
        
        return metrics, self.model_data