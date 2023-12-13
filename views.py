import pickle
import sys
from flask import Blueprint, Response, jsonify, render_template, request
import json
import os
import numpy as np
import pandas as pd
import heapq
from scipy import sparse

from index import construct_inverted_index, save_index
from vectorizer import calculate_tf_idf, save_tf_idf
from results import get_results, get_result
import time

DATA_FILE = 'Greek_Parliament_Proceedings_1989_2020.csv'
# DATA_FILE = 'Greek_Parliament_Proceedings_1989_2020_sample.csv'
# DATA_FILE = 'sample_data.csv'
# DATA_FILE = 'test_data.csv'
INDEX_FILE = 'inverted_index.json'
TFIDF_FILE = 'tfidf_index.json'
TFIDF_FILE = 'tfidf_matrix.npz'
TFIDF_VOCAB_FILE = 'tfidf_vocab.npz'
TFIDF_VEC_FILE = 'tfidf_vect.pkl'
DATA_FOLDER = 'data'

NO_KEYWORDS = 10

# Get the data file path
current_path = os.getcwd()
csv_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), DATA_FILE)
index_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), INDEX_FILE)
tfidf_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), TFIDF_FILE)
tfidf_vocab_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), TFIDF_VOCAB_FILE)
tfidf_vectorizer_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), TFIDF_VEC_FILE)

content_type='application/json; charset=utf-8'
views = Blueprint(__name__, "views")

@views.route('/', methods=['GET'])
def index():
    query = request.args.get('search-input', type=str)
    if not query:
        return render_template('index.html')
    start = time.time()
    results = get_results(query, csv_file_path, tfidf_vectorizer_file_path, tfidf_file_path)
    end = time.time()
    print('Results time: ', (end - start), ' sec(s)', file=sys.stderr)
    return render_template('results.html', query=query, results=results)

@views.route('/result', methods=['GET'])
def show_result():
    try:
        # Get the query from the URL parameters
        doc = request.args.get('doc')
        if not doc:
            raise ValueError('Query parameter is missing.')
        result = get_result(doc, csv_file_path)
        return render_template('result.html', result=result)
    except Exception as e:
        return jsonify({"error": str(e)})

@views.route('/get_index')
def get_index():
    # Load the inverted index from the JSON file
    if not os.path.exists(index_file_path):
        inverted_index = construct_inverted_index(csv_file_path)
        save_index(inverted_index, index_file_path)

    with open(index_file_path, 'r', encoding='utf-8') as json_file:
        inverted_index = json.load(json_file)

    # Serialize the data using json.dumps to avoid Unicode escape sequences
    response = Response(json.dumps(inverted_index, ensure_ascii=False), content_type=content_type)
    return response

@views.route('/get_tfidf')
def get_tftidf():
    # Load the tf-idf index from the JSON file
    if not os.path.exists(tfidf_file_path):
        start = time.time()
        tfidf_measurements = calculate_tf_idf(csv_file_path)
        end = time.time()
        print('TF-IDF calculation time: ', (end - start), ' sec(s)', file=sys.stderr)
        start = time.time()
        save_tf_idf(tfidf_measurements, tfidf_file_path, tfidf_vocab_file_path, tfidf_vectorizer_file_path)
        end = time.time()
        print('TF-IDF files saving calculation time: ', (end - start), ' sec(s)', file=sys.stderr)
    
    a = {"status": "finished"}
    response = Response(json.dumps(a, ensure_ascii=False), content_type=content_type)
    return response

def update_heap(heap, new_item):
    if len(heap) < NO_KEYWORDS:
        heapq.heappush(heap, new_item)
    elif new_item > heap[0]:
        heapq.heapreplace(heap, new_item)
    
def extract_top_keywords(heaps):
    for property_name, heap in heaps.items():
        top_keywords_per_property = [word for _, word in sorted(heap, reverse=True)]
        heaps[property_name] = top_keywords_per_property

@views.route('/top_keywords')
def get_top_keywords():
    # Initialize data structures to store top speeches, top speeches per member, and top speeches per political party
    top_speeches = []
    top_per_member = {}
    top_per_political_party = {}
    top_keywords = {"top_speeches": top_speeches, "top_speeches_per_member": top_per_member, "top_speeches_political_party": top_per_political_party}

    # Load the TF-IDF matrix and the feature names from the vocabulary file
    tfidf_matrix = sparse.load_npz(tfidf_file_path)
    with open(tfidf_vocab_file_path, 'rb') as file:
        feature_names = pickle.load(file)

    # Iterate over the CSV data in chunks
    for doc_idx, df_chunk in enumerate(pd.read_csv(csv_file_path, chunksize=1000, header=None)):
        # Extract speeches, member names, and political parties from the chunk
        speeches = df_chunk[10].tolist()
        member_names = df_chunk[0].tolist()
        political_parties = df_chunk[5].tolist()

        no_speeches = len(speeches) - 1

        # Iterate over each speech in the chunk
        for doc_idx in range(no_speeches):
            # Extract TF-IDF values for the current speech and get the indices of the top keywords based on TF-IDF values
            tfidf_values = tfidf_matrix[doc_idx].toarray()[0]
            top_indices = tfidf_values.argsort()[-np.count_nonzero(tfidf_values):][::-1] if np.count_nonzero(tfidf_values) < NO_KEYWORDS else tfidf_values.argsort()[-NO_KEYWORDS:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            
            # Update the list of top speeches
            top_speeches.append({"speech_id": doc_idx + 1, "top_words": top_words})
            
            # Extract the political party for the current speech
            member_name = member_names[doc_idx + 1]
            political_party = political_parties[doc_idx + 1]

            # Update the max heap for the current member and the current political party
            if member_name not in top_per_member:
                top_per_member[member_name] = []
            if political_party not in top_per_political_party:
                top_per_political_party[political_party] = []
            
            # Update the max heap for the current member and political party
            for top_id in top_indices:
                update_heap(top_per_member[member_name], (tfidf_values[top_id], feature_names[top_id]))
                update_heap(top_per_political_party[political_party], (tfidf_values[top_id], feature_names[top_id]))
            
    # Extract the top keywords from the max heap for each member and political party
    extract_top_keywords(top_per_member)
    extract_top_keywords(top_per_political_party)

    # Create a JSON response with the top speeches and keywords
    response = Response(json.dumps(top_keywords, ensure_ascii=False), content_type=content_type)
    return response
