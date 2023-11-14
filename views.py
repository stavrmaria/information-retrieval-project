from flask import Blueprint, Response, jsonify, render_template, request
import json
import os
from index import construct_inverted_index, save_index, process_text
from vectorizer import calculate_tf_idf, save_tf_idf

DATA_FILE = 'sample_data.csv'
INDEX_FILE = 'inverted_index.json'
TFIDF_FILE = 'tf_idf_index.json'
DATA_FOLDER = 'data'

# Get the data file path
current_path = os.getcwd()
csv_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), DATA_FILE)
index_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), INDEX_FILE)
tfidf_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), TFIDF_FILE)

content_type='application/json; charset=utf-8'
views = Blueprint(__name__, "views")

@views.route('/')
def index():
    return render_template('index.html')

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
    # Load the inverted index from the JSON file
    if not os.path.exists(index_file_path):
        get_index()
    
    # Load the tf-idf index from the JSON file
    if not os.path.exists(tfidf_file_path):
        tf_idf_index = calculate_tf_idf(csv_file_path)
        save_tf_idf(tf_idf_index, tfidf_file_path)

    with open(tfidf_file_path, 'r', encoding='utf-8') as json_file:
        tf_idf_index = json.load(json_file)

    # Serialize the data using json.dumps to avoid Unicode escape sequences
    response = Response(json.dumps(tf_idf_index, ensure_ascii=False), content_type=content_type)
    return response

@views.route('/query', methods=['GET'])
def query():
    try:
        # Get the query from the URL parameters
        user_query = request.args.get('q')
        if not user_query:
            raise ValueError('Query parameter is missing.')

        processed_query = process_text(user_query)
        response = Response(json.dumps(processed_query, ensure_ascii=False), content_type=content_type)
        return response
    except Exception as e:
        return jsonify({"error": str(e)})