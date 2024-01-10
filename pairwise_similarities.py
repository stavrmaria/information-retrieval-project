import pickle
import pandas as pd
from itertools import permutations
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_top_k_pairwise_similarities(csv_file_path, tfidf_vectorizer_file_path, k, threshold):
    # Load the speeches data from the data file
    speeches_data = pd.read_csv(csv_file_path, usecols=['member_name', 'speech'])

    # Load the TF-IDF matrix and vectorizer from the files
    with open(tfidf_vectorizer_file_path, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
    
    # Contruct the member Feature Vector
    members = speeches_data['member_name'].unique()
    member_feature_vectors = {}

    for member in members:
        member_speeches = speeches_data.loc[speeches_data['member_name'] == member, 'speech']
        member_tfidf_values = tfidf_vectorizer.transform(member_speeches)
        member_feature_vector = member_tfidf_values.mean(axis=0)
        member_feature_vectors[member] = member_feature_vector
    
    # Pairwise Similarity Calculation
    similar_pairs = []
    for pair in permutations(members, 2):
        member1, member2 = pair
        similarity_score = cosine_similarity(np.asarray(member_feature_vectors[member1]), np.asarray(member_feature_vectors[member2]))[0][0]
        if similarity_score >= threshold: similar_pairs.append((member1, member2, similarity_score))
    
    top_k_pairs = sorted(similar_pairs, key=lambda x: x[2], reverse=True)[:k]

    # Return the top-k pairs as a list of tuples
    return top_k_pairs
