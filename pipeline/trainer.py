from data_loader.data_loader import DataLoader

from featureselector.featureselector import FeatureSelector

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize, StandardScaler 

from trainers.dbscan import DBScan
from trainers.isolationforest import IsoForest

from utils.engineer_functions import temp_new_text


class Trainer:
    def __init__(self, **kwargs):
        valid_keys = ["config", "comet_config", "model_config", "experiment", "model", "tfidf"]
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

        print("Normalizing Data...")
        self.modelling_data = self.normalize_data()

        if self.model == "dbscan":
            metrics  = self.experiment_dbscan()
        elif self.model =="isolation_forest":
            metrics = self.experiment_isolation_forest()
            
        print("Saving results...")
        self.save_results(metrics)

    def experiment_params(self,params):
        self.experiment.log_parameters(params)

    def save_results(self,metrics):
        self.experiment.log_metrics(metrics)
        
        if self.model == "dbscan":
            if self.tfidf == 'y':
                self.model_data.to_csv(self.model_config.dbscan.results.tfidf_save_data_path, index=False)
                self.experiment.log_model(name=self.model,
                            file_or_folder=self.model_config.dbscan.results.tfidf_save_data_path)
            else:
                self.model_data.to_csv(self.model_config.dbscan.results.no_tfidf_save_data_path, index=False)
                self.experiment.log_model(name=self.model,
                            file_or_folder=self.model_config.dbscan.results.no_tfidf_save_data_path)
        elif self.model == "isolation_forest":
            if self.tfidf == 'y':
                self.model_data.to_csv(self.model_config.isolation_forest.results.tfidf_save_data_path, index=False)
                self.experiment.log_model(name=self.model,
                            file_or_folder=self.model_config.isolation_forest.results.tfidf_save_data_path)
            else:
                self.model_data.to_csv(self.model_config.isolation_forest.results.no_tfidf_save_data_path, index=False)
                self.experiment.log_model(name=self.model,
                            file_or_folder=self.model_config.isolation_forest.results.no_tfidf_save_data_path)
        
    def experiment_dbscan(self):
        print("Loading DBScan...")
        self.trainer = DBScan(model_config = self.model_config, model_df = self.modelling_data)
        params = self.trainer.hypertune_dbscan_params()

        print("Parsing parameters to Experiment...\nTesting parameters: {}".format(params))
        self.experiment_params(params)

        results = self.trainer.dbscan_cluster(params)

        print("Evaluating DBScan...")
        silhouette_avg = silhouette_score(self.modelling_data, results)

        self.model_data['dbscan_clusters'] = results
        self.model_data['fake_reviews'] = [1 if x == -1 else 0 for x in results]

        total_reviews = len(list(results))
        total_fake_reviews = list(results).count(-1)
        total_non_fake_reviews = total_reviews - total_fake_reviews

        metrics = {"silhouette_avg":silhouette_avg,"total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}
        
        return metrics

    def experiment_isolation_forest(self):
        print("Loading Isolation Forest...")
        self.trainer = IsoForest(model_config = self.model_config, model_df = self.modelling_data)
        params = self.trainer.make_isolation_forest()

        print("Parsing parameters to Experiment...\nTesting parameters: {}".format(params))
        self.experiment_params(params)

        results = self.trainer.predict_anomalies()

        print("Evaluating Isolation Forest...")

        self.model_data['isolation_forest'] = results
        self.model_data['fake_reviews'] = [1 if x == -1 else 0 for x in results]

        total_reviews = len(list(results))
        total_fake_reviews = list(results).count(-1)
        total_non_fake_reviews = total_reviews - total_fake_reviews

        metrics = {"total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}
        
        return metrics
