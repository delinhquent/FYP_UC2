import math

import numpy as np

from kneed import KneeLocator

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors

import tqdm


class DBScan:
    def __init__(self, model_config, model_df):
        self.model_config = model_config
        self.model_df = model_df
        interested_columns = [column for column in self.model_df.columns if column != "manual_label"]
        self.modelling_data = self.model_df[interested_columns].values

    def hypertune_dbscan_params(self):
        print("Finding optimal epsilon and min_sample value for DBScan Model...")

        print("Finding no. clusters by KMeans...")
        cluster = self.find_optimized_cluster()

        print("Finding optimzied epsilon value...")
        eps = self.find_optimized_eps(cluster)

        print("Finding optimzied min_sample value...")
        min_sample = 2*len(self.modelling_data.columns)

        params = {"eps":eps,"min_samples":min_sample}

        return params

    def evaluate_dbscan(self, results):
        print("Evaluating DBScan...")
        silhouette_avg = silhouette_score(self.modelling_data, results)

        total_reviews = len(list(results))
        total_fake_reviews = list(results).count(-1)
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

        metrics = {"f1":f1Score,"precision":precScore,"recall":recallScore,"silhouette_avg":silhouette_avg, "total_fake_reviews": total_fake_reviews,"percentage_fake_reviews": (total_fake_reviews/total_reviews),"total_non_fake_reviews":total_non_fake_reviews,"percentage_non_fake_reviews":total_non_fake_reviews/total_reviews}
        
        return metrics
        
    def dbscan_cluster(self,params):
        print("Performing DBScan...")
        dbscan_model = DBSCAN(eps=params['eps'],min_samples=params['min_samples']).fit(self.modelling_data)
        core_samples_mask = np.zeros_like(dbscan_model.labels_,dtype=bool)
        core_samples_mask[dbscan_model.core_sample_indices_] = True
        labels = dbscan_model.labels_

        return labels

    def find_optimized_cluster(self):
        max_range = self.model_config.dbscan.hyperparam_test.range_n_clusters + 1
        selected_random_state = self.model_config.dbscan.hyperparam_test.random_state

        range_n_clusters = range(3,max_range)
        kmeans_silhouette_results = {}

        for n_clusters in tqdm.tqdm(range_n_clusters):
            clusterer = KMeans(n_clusters=n_clusters, random_state = selected_random_state)
            cluster_labels = clusterer.fit_predict(self.modelling_data)

            silhouette_avg = silhouette_score(self.modelling_data, cluster_labels)
            kmeans_silhouette_results[n_clusters] = silhouette_avg
            print("For n_clusters = {}, the average silhouette_score is : {}".format(n_clusters, silhouette_avg))

        chosen_cluster = max(kmeans_silhouette_results, key=kmeans_silhouette_results.get)
        print("Chosen cluster by K-Means is {}. Its average silhouette score is {}\n".format(chosen_cluster,kmeans_silhouette_results[chosen_cluster]))
        return chosen_cluster
    
    def find_optimized_eps(self, cluster):
        nearest_neighbors = NearestNeighbors(n_neighbors=cluster)
        neighbors = nearest_neighbors.fit(self.modelling_data)
        distances, indices = neighbors.kneighbors(self.modelling_data)

        distances = np.sort(distances[:,cluster-1], axis=0)

        i = np.arange(len(distances))
        knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

        chosen_eps = max(0.1,distances[knee.knee])

        print("Chosen eps is {}.\n".format(chosen_eps))
        return chosen_eps

