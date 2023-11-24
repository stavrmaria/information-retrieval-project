import sys
from flask import Blueprint, Response, jsonify, render_template, request
import json
import os

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
