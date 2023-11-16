import os
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pandas as pd
from index import process_text

def load_inverted_index(json_file, encoding='utf-8'):
    # Load the inverted index from the JSON file
    with open(json_file, 'r', encoding=encoding) as file:
        inverted_index = json.load(file)
    return dict(inverted_index)

# Apply TF-IDF weights to a document collection based on the inverted index stored in a JSON file
def calculate_tf_idf(data_file_path):
    # Read the speech column of the csv file
    df_speeches = pd.read_csv(data_file_path, usecols=['speech'])
    tokenized_speeches = df_speeches['speech'].apply(process_text)
    processed_texts = [" ".join(tokens) for tokens in tokenized_speeches]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()    
    document_ids = df_speeches.index
    
    # Create a dictionary to store the TF-IDF representation
    tfidf_dict = {}
    for term_idx, term in enumerate(feature_names):
        # Extract non-zero TF-IDF values for the term
        term_tfidf_values = tfidf_matrix[:, term_idx].data
        # Extract the row indices of the non-zero values
        nonzero_row_indices = tfidf_matrix[:, term_idx].nonzero()[0]

        doc_tfidf_dict = {str(nonzero_row_indices[i]): term_tfidf_values[i] for i in range(len(term_tfidf_values))} # {doc_id: tfidf_value}
        
        tfidf_dict[term] = doc_tfidf_dict
    
    return tfidf_dict

def save_tf_idf(tf_idf_index, json_file, encoding='utf-8'):
    with open(json_file, 'w', encoding=encoding) as f:
        json.dump(tf_idf_index, f, indent=4, ensure_ascii=False)
