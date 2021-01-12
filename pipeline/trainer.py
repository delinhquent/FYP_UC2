from data_loader.data_loader import DataLoader

from featureselector.featureselector import FeatureSelector

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize, StandardScaler 

from trainers.dbscan import DBScan
from trainers.isolationforest import IsoForest
from trainers.extendedisolationforest import EIF

from utils.engineer_functions import temp_new_text


class Trainer:
    def __init__(self, **kwargs):
        valid_keys = ["config", "comet_config", "model_config", "experiment", "model", "tfidf", "feature_select","normalize"]
        for key in valid_keys:
            setattr(self, key, kwargs.get(key))
        self.model_data_loader = DataLoader(self.config.model.save_data_path)
        self.model_data = None
        self.tfidf_data = None
        self.modelling_data = None
        self.trainer = None

    def load_data(self):
        self.model_data = self.get_model_data()

    def get_model_data(self):
        self.model_data_loader.load_data()
        return self.model_data_loader.get_data()
    
    def get_tfidf_vector(self):
        print("Generating TFIDF Vector...")
        vec = TfidfVectorizer (ngram_range = (1,2),min_df=0.1,max_df=0.9)
        reviews = temp_new_text(list(self.model_data['cleaned_reviews_text']))
        tfidf_reviews = vec.fit_transform(reviews)
        tfidf_reviews_df = pd.DataFrame(tfidf_reviews.toarray(), columns=vec.get_feature_names())
        
        return tfidf_reviews_df.fillna(value=0)
    
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
        if self.tfidf == 'y':
            self.tfidf_data = self.get_tfidf_vector()
            print("Combining vectors with dataset...")
            return pd.concat([modelling_df, self.tfidf_data],axis=1)
        else:
            return modelling_df

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

    def train_model(self):
        print("Retrieving necessary columns for modelling...")
        self.modelling_data = self.get_modelling_data()

        if self.normalize == 'y':
            print("Normalizing Data...")
            self.modelling_data = self.normalize_data()

        if self.model == "dbscan":
            metrics = self.dbscan_pipeline()
        elif self.model == "isolation_forest":
            metrics = self.isolation_forest_pipeline()
        elif self.model == "eif":
            metrics = self.eif_pipeline()
            
        print("Saving results...")
        self.save_results(metrics)

    def experiment_params(self,params):
        self.experiment.log_parameters(params)

    def save_results(self,metrics):
        self.experiment.log_metrics(metrics)
        
        results_path = {
            "dbscan" : self.model_config.dbscan.results.save_data_path,
            "isolation_forest": self.model_config.isolation_forest.results.save_data_path,
            "eif" : self.model_config.eif.results.save_data_path
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
        self.model_data['fake_reviews'] = [1 if x == -1 else 0 for x in results]

        metrics = self.trainer.evaluate_dbscan(results)
        
        return metrics
    
    def isolation_forest_pipeline(self):
        print("Loading Isolation Forest...")
        self.trainer = IsoForest(model_config = self.model_config, model_df = self.modelling_data)

        params = self.trainer.make_isolation_forest()

        print("Parsing parameters to Experiment...\nTesting parameters: {}".format(params))
        self.experiment_params(params)

        results = self.trainer.predict_anomalies()

        self.model_data['isolation_forest'] = results
        self.model_data['fake_reviews'] = [1 if x == -1 else 0 for x in results]

        metrics = self.trainer.evaluate_isolation_forest(results)
    
        return metrics

    def eif_pipeline(self):
        print("Loading Extended Isolation Forest...")
        self.trainer = EIF(model_config = self.model_config, model_df = self.modelling_data)
        
        params = self.trainer.make_eif()

        print("Parsing parameters to Experiment...\nTesting parameters: {}".format(params))
        self.experiment_params(params)

        results = self.trainer.predict_anomalies()

        self.model_data['eif'] = results
        self.model_data['fake_reviews'] = [1 if x == -1 else 0 for x in results]

        metrics = self.trainer.evaluate_eif(results)

        return metrics