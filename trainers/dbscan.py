import math

import numpy as np

from kneed import KneeLocator

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


class DBScan:
    def __init__(self, model_config, model_df):
        self.model_config = model_config
        self.model_df = model_df

    def hypertune_dbscan_params(self):
        print("Finding optimal epsilon and min_sample value for DBScan Model...")

        print("Finding no. clusters by KMeans...")
        cluster = self.find_optimized_cluster()

        print("Finding optimzied epsilon value...")
        eps = self.find_optimized_eps(cluster)

        print("Finding optimzied min_sample value...")
        min_sample = self.find_optimized_min_sample(eps,cluster)

        params = {"eps":eps,"min_samples":min_sample}

        return params
    
    def dbscan_cluster(self,params):
        print("Performing DBScan...")
        dbscan_model = DBSCAN(eps=params['eps'],min_samples=params['min_samples']).fit(self.model_df)
        core_samples_mask = np.zeros_like(dbscan_model.labels_,dtype=bool)
        core_samples_mask[dbscan_model.core_sample_indices_] = True
        labels = dbscan_model.labels_
        
        return labels

    def find_optimized_cluster(self):
        max_range = self.model_config.dbscan_hyperparam_test.range_n_clusters + 1
        selected_random_state = self.model_config.dbscan_hyperparam_test.random_state

        range_n_clusters = range(3,max_range)
        kmeans_silhouette_results = {}

        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state = selected_random_state)
            cluster_labels = clusterer.fit_predict(self.model_df)

            silhouette_avg = silhouette_score(self.model_df, cluster_labels)
            kmeans_silhouette_results[n_clusters] = silhouette_avg
            print("For n_clusters = {}, the average silhouette_score is : {}".format(n_clusters, silhouette_avg))

        chosen_cluster = max(kmeans_silhouette_results, key=kmeans_silhouette_results.get)
        print("Chosen cluster by K-Means is {}. Its average silhouette score is {}\n".format(chosen_cluster,kmeans_silhouette_results[chosen_cluster]))
        return chosen_cluster
    
    def find_optimized_eps(self, cluster):
        nearest_neighbors = NearestNeighbors(n_neighbors=cluster)
        neighbors = nearest_neighbors.fit(self.model_df)
        distances, indices = neighbors.kneighbors(self.model_df)

        distances = np.sort(distances[:,cluster-1], axis=0)

        i = np.arange(len(distances))
        knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

        chosen_eps = distances[knee.knee]

        print("Chosen eps is {}.\n".format(chosen_eps))
        return chosen_eps

    def find_optimized_min_sample(self, eps, kmeans_cluster):
        min_sample_value = 1
        clusters = 0

        while clusters != kmeans_cluster:
            dbscan_model = DBSCAN(eps=eps,min_samples=min_sample_value).fit(self.model_df)
            core_samples_mask = np.zeros_like(dbscan_model.labels_,dtype=bool)
            core_samples_mask[dbscan_model.core_sample_indices_] = True
            labels = set([label for label in dbscan_model.labels_ if label >=0])
            clusters = len(set(labels))
            print("For min_samples value = {}, total no. of clusters are : {}".format(min_sample_value, clusters))

            if clusters//kmeans_cluster >= 1.1:
                min_sample_value += max(math.floor(0.2*min_sample_value),self.model_config.dbscan_hyperparam_test.max_min_sample_increment)
            elif clusters//kmeans_cluster == 0:
                min_sample_value -= 1
            else:
                min_sample_value += 1

        return min_sample_value