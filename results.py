import pandas as pd
from index import process_text

import sys
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_results(query, data_file_path):
    query = process_text(query)

    print(query, file=sys.stderr)

    # Concatenate the strings with a space as a delimiter
    query = " ".join(query)

    # Load the vectorizer in another function or part of your code
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        loaded_vectorizer = pickle.load(file)

    # Use the loaded vectorizer to transform a new query
    query_tfidf = loaded_vectorizer.transform([query])

    # Load the matrix in a different function or part of your code
    # tfidf_matrix = np.load('tfidf_matrix.npy', allow_pickle=True)

    # Load the array from the file using pickle
    with open('matrix.pkl', 'rb') as file:
        tfidf_matrix = pickle.load(file)

    # print(query_tfidf.shape, file=sys.stderr)
    # print(tfidf_matrix.shape, file=sys.stderr)

    # # Calculate cosine similarity between the query and each document
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # # Display the cosine similarities for each document
    # for i, similarity in enumerate(cosine_similarities):
    #     print(f"Similarity with Document {i + 1}: {similarity}", file=sys.stderr)

    # print(query_tfidf.shape, file=sys.stderr)

    # Get indices of the 5 greatest values
    k = 6
    top_indices = np.argsort(cosine_similarities)[-k:][::-1]

    # print(top_indices, file=sys.stderr)

    df_speeches = pd.read_csv(data_file_path)

    results = []
    for i in range(k):
        doc_id=top_indices[i]
        # print(df_speeches.values[doc_id][10], file=sys.stderr)

        speech = df_speeches.values[doc_id][10]
        # Split the string into words
        words = speech.split()

        # Get the first 10 words
        first_10_words = words[:10]

        # Join the words back into a string
        result_string = ' '.join(first_10_words)

        if len(words)>10:
            result_string += "..." 

        results.append({"speech_start"  : result_string,
                       "name"          : df_speeches.values[doc_id][0],
                       "political_party" : df_speeches.values[doc_id][5],
                       "date"          : df_speeches.values[doc_id][1],
                       "score"         : cosine_similarities[doc_id],
                       "doc_id"         : doc_id})
    return results

def get_result(doc_id, data_file_path):
    result = []

    df_speeches = pd.read_csv(data_file_path)

    doc_id = int(doc_id)
    result.append(df_speeches.values[doc_id][0]) # name
    result.append(df_speeches.values[doc_id][1]) # date
    result.append(df_speeches.values[doc_id][5]) # political party
    result.append(df_speeches.values[doc_id][10]) # speech

    return result