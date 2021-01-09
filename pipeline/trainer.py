from data_loader.data_loader import DataLoader

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from trainers.dbscan import DBScan

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
        if self.tfidf == 'y':
            self.tfidf_data = self.get_tfidf_vector()
            print("Combining vectors with dataset...")
            return pd.concat([modelling_df, self.tfidf_data],axis=1)
        else:
            return modelling_df

    def remerge_data(self):
        return pd.concat([self.model_data[['asin','acc_num','cleaned_reviews_profile_link']],self.modelling_data],axis=1)

    def experiment_dbscan(self):
        print("Retrieving necessary columns for modelling...")
        self.modelling_data = self.get_modelling_data()

        print("Loading DBScan...")
        self.trainer = DBScan(model_config = self.model_config, model_df = self.modelling_data)
        params = self.trainer.hypertune_dbscan_params()

        print("Parsing parameters to Experiment...\nTesting parameters: {}".format(params))
        self.experiment_params(params)

        results = self.trainer.dbscan_cluster(params)

        print("Evaluating DBScan...")
        silhouette_avg = silhouette_score(self.modelling_data, results)

        self.modelling_data['dbscan_clusters'] = results
        self.modelling_data['fake_reviews'] = [1 if x == -1 else 0 for x in results]
        self.modelling_data = self.remerge_data()

        total_reviews = len(list(results))
        total_fake_reviews = list(results).count(-1)
        total_non_fake_reviews = total_reviews - total_fake_reviews

        metrics = {"silhouette_avg":silhouette_avg,"total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}
        
        print("Saving results...")
        self.save_results(metrics)

    def experiment_params(self,params):
        self.experiment.log_parameters(params)

    def save_results(self,metrics):
        self.experiment.log_metrics(metrics)

        if self.tfidf == 'y':
            self.modelling_data.to_csv(self.model_config.dbscan_results.tfidf_save_data_path, index=False)
            self.experiment.log_model(name=self.model,
                         file_or_folder=self.model_config.dbscan_results.tfidf_save_data_path)
        else:
            self.modelling_data.to_csv(self.model_config.dbscan_results.no_tfidf_save_data_path, index=False)
            self.experiment.log_model(name=self.model,
                         file_or_folder=self.model_config.dbscan_results.no_tfidf_save_data_path)
        

