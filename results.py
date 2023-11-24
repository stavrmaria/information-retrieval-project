import pandas as pd
from scipy import sparse
from index import process_text

import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

UPPER_WORDS_LIMIT = 10

# Function to get search results based on a query
def get_results(query, data_file_path, tfidf_vectorizer_file_path, tfidf_file_path):
    query = process_text(query)

    # Load the TF-IDF matrix and vectorizer from the files
    tfidf_matrix = sparse.load_npz(tfidf_file_path)
    with open(tfidf_vectorizer_file_path, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)

    # Use the loaded vectorizer to transform the new query
    query_tfidf = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Get indices of the 5 greatest values
    k = 6
    top_indices = np.argsort(cosine_similarities)[-k:][::-1]
    df_speeches = pd.read_csv(data_file_path)
    results = []
    for i in range(k):
        doc_id = top_indices[i]
        speech = df_speeches['speech'].iloc[doc_id]
        words = speech.split()
        first_words = words
        if len(words) > UPPER_WORDS_LIMIT:
            first_words = words[:UPPER_WORDS_LIMIT]
        result_string = ' '.join(first_words)

        if len(words) > UPPER_WORDS_LIMIT:
            result_string += "..."

        results.append({
            "speech_start": result_string,
            "name": df_speeches.values[doc_id][0],
            "political_party": df_speeches.values[doc_id][5],
            "date": df_speeches.values[doc_id][1],
            "score": cosine_similarities[doc_id],
            "doc_id": doc_id
        })
    
    return results

def get_result(doc_id, data_file_path):
    result = []
    df_speeches = pd.read_csv(data_file_path)
    doc_id = int(doc_id)
    for i in range(11):
        result.append(df_speeches.values[doc_id][i])

    return result