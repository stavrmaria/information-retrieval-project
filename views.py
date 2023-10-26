from flask import Blueprint, Response, render_template
import json
import os
from index import construct_inverted_index, save_index

DATA_FILE = 'sample_data.csv'
INDEX_FILE = 'inverted_index.json'
DATA_FOLDER = 'data'

# Get the data file path
current_path = os.getcwd()
csv_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), DATA_FILE)
views = Blueprint(__name__, "views")

@views.route('/')
def index():
    return render_template('index.html')

@views.route('/get_index')
def get_index():
    # Load the inverted index from the JSON file
    index_file_path = os.path.join(os.getcwd(), INDEX_FILE)
    if not os.path.exists(index_file_path):
        inverted_index = construct_inverted_index(csv_file_path)
        save_index(inverted_index, INDEX_FILE)

    with open('inverted_index.json', 'r', encoding='utf-8') as json_file:
        inverted_index = json.load(json_file)

    # Serialize the data using json.dumps to avoid Unicode escape sequences
    response = Response(json.dumps(inverted_index, ensure_ascii=False), content_type='application/json; charset=utf-8')
    return response