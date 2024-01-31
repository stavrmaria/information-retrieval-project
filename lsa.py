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


def get_topics(tfidf_file_path, tfidf_vectorizer_file_path):

    # Load the TF-IDF matrix and vectorizer from the files
    tfidf_matrix = sparse.load_npz(tfidf_file_path)

    # Define the number of topics or components
    num_components=10

    # Create SVD object
    lsa = TruncatedSVD(n_components=num_components, n_iter=100, random_state=42)

    # Fit SVD model on data
    lsa.fit_transform(tfidf_matrix)

    # Get Singular values and Components 
    Sigma = lsa.singular_values_ 
    V_transpose = lsa.components_.T

    # ----------------------------------------------------------------------------------
    
    # Load the TF-IDF matrix and vectorizer from the files
    with open(tfidf_vectorizer_file_path, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)

    # Print the topics with their terms
    terms = tfidf_vectorizer.get_feature_names_out()

    # Initialize a list to store the top terms for each topic
    topics = []

    for index, component in enumerate(lsa.components_):
        zipped = zip(terms, component)

        # Sort the terms based on their weights in descending order and select the top 10 terms for the current topic.
        top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:10]

        top_terms_list=list(dict(top_terms_key).keys())

        topics.append(top_terms_list)

        print("Topic "+str(index)+": ",top_terms_list, file=sys.stderr)
    
    return topics