import itertools
import multiprocessing
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import sys

def calculate_similarity(pair, member_feature_vectors, threshold):
    member1, member2 = pair
    similarity_score = cosine_similarity(np.asarray(member_feature_vectors[member1]), np.asarray(member_feature_vectors[member2]))[0][0]
    if similarity_score >= threshold:
        return (member1, member2, similarity_score)
    else:
        return None

def calculate_similarity_wrapper(args):
    return calculate_similarity(*args)

def process_chunk(chunk, tfidf_vectorizer):
    member_feature_vectors = {}
    # Extract unique members in the current chunk
    for member, group in chunk.groupby('member_name'):
        member_tfidf_values = tfidf_vectorizer.transform(group['speech'])
        member_feature_vector = member_tfidf_values.mean(axis=0)
        member_feature_vectors[member] = member_feature_vector
    return member_feature_vectors

def process_chunk_wrapper(args):
    return process_chunk(*args)

def get_top_k_pairwise_similarities(csv_file_path, tfidf_vectorizer_file_path, k, threshold):
    # Load the TF-IDF matrix and vectorizer from the files
    with open(tfidf_vectorizer_file_path, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)

    # Load the speeches data from the data file
    chunksize = 100000
    speeches_data = pd.read_csv(csv_file_path, usecols=['member_name', 'speech'], chunksize=chunksize)
    
    # Contruct the member Feature Vector
    member_feature_vectors = {}
    with ProcessPoolExecutor() as executor:
        chunk_tfidf_vectorizer = [(chunk, tfidf_vectorizer) for chunk in speeches_data]
        results = executor.map(process_chunk_wrapper, chunk_tfidf_vectorizer)
    
    for result in results:
        for member, vector in result.items():
            if member in member_feature_vectors:
                member_feature_vectors[member] += vector
            else:
                member_feature_vectors[member] = vector
    
    members = list(member_feature_vectors.keys())
    
    with multiprocessing.Pool() as pool:
        similarity_scores = pool.starmap(calculate_similarity, 
                                         ((pair, member_feature_vectors, threshold) for pair in itertools.permutations(members, 2)))
    similar_pairs = [result for result in similarity_scores if result is not None]

    top_k_pairs = sorted(similar_pairs, key=lambda x: x[2], reverse=True)[:k]

    return top_k_pairs
