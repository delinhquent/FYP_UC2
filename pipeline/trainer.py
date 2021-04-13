from data_loader.data_loader import DataLoader
from featureselector.featureselector import FeatureSelector
from impactscorer.impactscorer import ImpactScorer
from trainers.models import Model

import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
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
        self.model_data = self.model_data[(self.model_data['acc_num'].notnull()) & (self.model_data['cleaned_reviews_text'].notnull())]
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

        unnessary_columns = ['asin','acc_num','cleaned_reviews_profile_link','decoded_comment','cleaned_reviews_text','cleaned_reviews_date_posted','cleaned_reviews_location','locale','manual_label']
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
        print(self.modelling_data.shape)


    def normalize_data(self,modelling_df):
        scaler = StandardScaler() 
        X_scaled = scaler.fit_transform(modelling_df) 
        pickle.dump(scaler, open('models/normalizer/feature_normalizer_standard.pkl','wb'))
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

        metrics, self.modelling_data = self.model_pipeline()
        self.model_data['fake_reviews'] = self.modelling_data['fake_reviews']
        self.model_data['decision_function'] = self.modelling_data['decision_function']

        print("Assessing impact...")
        assessor = ImpactScorer(self.model_data,self.profile_data)

        self.model_data, self.profile_data = assessor.assess_impact()

        print("Saving results...")
        self.save_results(metrics)
        self.profile_data.to_csv(self.config.profiles.save_data_path,index=False)

        print("Training of Model Completed...")

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
                        

    def model_pipeline(self):
        name_dict = {"ocsvm":"One-Class SVM",
            "copod":"Copula Based Outlier Detector", "hbos": "Histogram-based Outlier Detection",
            "pyod_isolation_forest": "Isolation Forest (PYoD)", "isolation_forest": "Isolation Forest",
            "pyod_lof": "Local Outlier Factor (PYoD)", "lof": "Local Outlier Factor"}
        print("Loading {}...".format(name_dict[self.model]))
        self.trainer = Model(model_config = self.model_config, model_df = self.modelling_data)

        params = self.trainer.make_model(self.model)

        print("Parsing parameters to Experiment...\nTesting parameters: {}".format(params))
        self.experiment_params(params)

        self.trainer.train_model()

        metrics, self.modelling_data = self.trainer.evaluate_model(name_dict[self.model],self.model)
        
        return metrics, self.modelling_data
