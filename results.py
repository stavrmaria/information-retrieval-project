import sys
import pandas as pd
from scipy import sparse
from vectorizer import process_text

import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

UPPER_WORDS_LIMIT = 10
NO_RESULTS = 10

def read_csv(data_file_path, chunksize=1000):
    for chunk in pd.read_csv(data_file_path, chunksize=chunksize):
        yield chunk

# Function to get search results based on a query

def get_results(query, data_file_path, tfidf_vectorizer_file_path, tfidf_file_path, score_threshold = 0.001):
    query = process_text(query)

    # Load the TF-IDF matrix and vectorizer from the files
    tfidf_matrix = sparse.load_npz(tfidf_file_path)
    with open(tfidf_vectorizer_file_path, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)

    # Use the loaded vectorizer to transform the new query
    query_tfidf = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Filter out top indices with scores below the threshold
    top_indices = np.argsort(cosine_similarities)[-NO_RESULTS:][::-1]
    top_indices = [index for index in top_indices if cosine_similarities[index] >= score_threshold]
    if all(score < score_threshold for score in cosine_similarities[top_indices]):
        return []
    
    results = []
    remaining_top_indices = set(top_indices)

    # Iterate through chunks of the CSV file
    for df_chunk in read_csv(data_file_path):
        chunk_results = []
        # Filter rows based on top indices
        df_chunk = df_chunk[df_chunk.index.isin(remaining_top_indices)]

        # Iterate through the rows in the current chunk
        for i, row in df_chunk.iterrows():
            speech = row['speech']
            words = speech.split()
            first_words = words if len(words) <= UPPER_WORDS_LIMIT else words[:UPPER_WORDS_LIMIT]
            result_string = ' '.join(first_words) + ("..." if len(words) > UPPER_WORDS_LIMIT else "")
            chunk_results.append({
                "speech_start": result_string,
                "name": row['member_name'],
                "political_party": row['political_party'],
                "date": row['sitting_date'],
                "score": cosine_similarities[i],
                "doc_id": i
            })
            remaining_top_indices.remove(i)

        results.extend(chunk_results)
        if not remaining_top_indices:
            break

    results = sorted(results, key=lambda x: x['score'], reverse=True)
    return results

def get_result(doc_id, data_file_path):
    result = []
    df_speeches = pd.read_csv(data_file_path)
    doc_id = int(doc_id)
    for i in range(11):
        result.append(df_speeches.values[doc_id][i])

    return result