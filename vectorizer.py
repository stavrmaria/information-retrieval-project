import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pandas as pd
from index import process_text
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from scipy import sparse
import pickle

def load_inverted_index(json_file, encoding='utf-8'):
    # Load the inverted index from the JSON file
    with open(json_file, 'r', encoding=encoding) as file:
        inverted_index = json.load(file)
    return dict(inverted_index)

def process_text_parallel(chunk):
    return chunk.apply(process_text)

def calculate_tf_idf(data_file_path):
    chunksize = 100000
    df_chunks = pd.read_csv(data_file_path, usecols=['speech'], chunksize=chunksize)
    df = pd.concat(df_chunks)

    num_cores = multiprocessing.cpu_count()
    max_workers= num_cores // 2

    # Process text in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        processed_text = list(executor.map(process_text_parallel, np.array_split(df['speech'], len(df))))
    processed_text = pd.concat(processed_text)

    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.9,
        max_features=6000
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_text)
    tfidf_vocab = tfidf_vectorizer.get_feature_names_out()

    return tfidf_matrix, tfidf_vectorizer, tfidf_vocab

def save_tf_idf(tfidf_measurements, tfidf_file_path, tfidf_vocab_file_path, tfidf_vectorizer_file_path):
    # Save the TF-IDF matrix in CSR format using scipy
    sparse.save_npz(tfidf_file_path, tfidf_measurements[0])
    # Save the fitted vectorizer to a file
    with open(tfidf_vectorizer_file_path, 'wb') as file:
        pickle.dump(tfidf_measurements[1], file)
    # Save the corresponding vocabulary (features) 
    with open(tfidf_vocab_file_path, 'wb') as file:
        pickle.dump(tfidf_measurements[2], file)

