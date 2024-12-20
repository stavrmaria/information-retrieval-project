import pickle
import pandas as pd
from itertools import permutations
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import os
from sklearn.decomposition import TruncatedSVD

import sys
from scipy import sparse
import pickle

from sklearn.cluster import KMeans

def get_matrix_k(tfidf_file_path):

    # Load the TF-IDF matrix and vectorizer from the files
    tfidf_matrix = sparse.load_npz(tfidf_file_path)

    # Define the number of topics or components
    num_components=10

    # Create SVD object
    lsa = TruncatedSVD(n_components=num_components, n_iter=100, random_state=42)

    # Fit SVD model on data
    matrix_k = lsa.fit_transform(tfidf_matrix)

    return matrix_k

def get_clusters(matrix_k, csv_file_path, num_clusters):

    df = pd.read_csv(csv_file_path)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(matrix_k)

    # # Print the first 10 speeches in cluster 2
    # cluster_to_print = 2
    # speeches_in_cluster = df.loc[cluster_labels == (cluster_to_print - 1), df.columns[10]].head(10)

    # print(f"First 10 speeches in Cluster {cluster_to_print}:\n")
    # for idx, speech in enumerate(speeches_in_cluster):
    #     print(f"Speech {idx + 1}:\n{speech}\n")

    # return

    # Initialize a list to store the structured representation of clusters
    structured_clusters = []

    for cluster_id in range(num_clusters):
        # Get indices of speeches in the current cluster
        speeches_indices = df.index[cluster_labels == cluster_id]

        # Extract the first 10 speeches in the current cluster
        speeches_in_cluster = df.loc[speeches_indices, df.columns[10]].head(10).tolist()

        # Append the structured representation of the cluster to the list
        structured_cluster = {
            "cluster_id": cluster_id,
            "speeches": speeches_in_cluster
        }

        structured_clusters.append(structured_cluster)

        print(f"Cluster {cluster_id}:\n{speeches_in_cluster}\n")

    return structured_clusters