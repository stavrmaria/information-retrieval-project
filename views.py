import sys
from flask import Blueprint, Response, jsonify, render_template, request
import json
import os
import time
import plotly.graph_objects as go

from index import construct_inverted_index, save_index
from vectorizer import calculate_tf_idf, save_tf_idf
from results import get_results, get_result
from plotter import extract_dates_top_words_per_member, get_top_keywords, extract_dates_top_words_per_party, extract_dates_top_words_per_speech
from pairwise_similarities import get_top_k_pairwise_similarities, construct_ps_graph
from lsa import get_topics
from clustering import get_matrix_k, get_clusters
from reader import create_files

INPUT_DATA_FILE = 'Greek_Parliament_Proceedings_1989_2020.csv'
INPUT_DATA_FILE = 'sample_data.csv'
DATA_FILE = 'output_file.csv'
SAMPLE_DATA_FILE = 'output_sample.csv'

INDEX_FILE = 'inverted_index.json'
TFIDF_FILE = 'tfidf_index.json'
TFIDF_FILE = 'tfidf_matrix.npz'
TFIDF_VOCAB_FILE = 'tfidf_vocab.npz'
TFIDF_VEC_FILE = 'tfidf_vect.pkl'
TOP_KEYWORDS_FILE = 'top_keywords.json'
DATA_FOLDER = 'data'
TEMPLATES_FOLDER = 'templates'
MEMBER_PLOT_PATH = 'top_keywords_member_plot.html'
PARTY_PLOT_PATH = 'top_keywords_party_plot.html'
SPEECH_PLOT_PATH = 'top_keywords_speech_plot.html'

TFIDF_SAMPLE_FILE = 'tfidf_matrix_sample.npz'
TFIDF_VOCAB_SAMPLE_FILE = 'tfidf_vocab_sample.npz'
TFIDF_VEC_SAMPLE_FILE = 'tfidf_vect_sample.pkl'

NO_KEYWORDS = 10
SIMILARITY_THRESHOLD = 0.6
FRACTION = 0.1
ROWS_LIMIT = 1000
CLUSTERING_LIMIT = 20

# Get the data file path
current_path = os.getcwd()
csv_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), DATA_FILE)
csv_sample_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), SAMPLE_DATA_FILE)
index_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), INDEX_FILE)
tfidf_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), TFIDF_FILE)
tfidf_vocab_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), TFIDF_VOCAB_FILE)
tfidf_vectorizer_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), TFIDF_VEC_FILE)
top_keywords_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), TOP_KEYWORDS_FILE)
member_plot_html_path = os.path.join(os.path.join(os.getcwd(), TEMPLATES_FOLDER), MEMBER_PLOT_PATH)
party_plot_html_path = os.path.join(os.path.join(os.getcwd(), TEMPLATES_FOLDER), PARTY_PLOT_PATH)
speech_plot_html_path = os.path.join(os.path.join(os.getcwd(), TEMPLATES_FOLDER), SPEECH_PLOT_PATH)

tfidf_sample_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), TFIDF_SAMPLE_FILE)
tfidf_vocab_sample_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), TFIDF_VOCAB_SAMPLE_FILE)
tfidf_vectorizer_sample_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), TFIDF_VEC_SAMPLE_FILE)

no_data_rows = 0
no_sample_rows = 0

content_type='application/json; charset=utf-8'
views = Blueprint(__name__, "views")

@views.route('/process_data')
def process_data():
    global no_data_rows, no_sample_rows
    no_data_rows, no_sample_rows = create_files(DATA_FOLDER, INPUT_DATA_FILE, DATA_FILE, SAMPLE_DATA_FILE, FRACTION)
    print("No Rows (process_data) = ", no_data_rows, file=sys.stderr)
    a = {"status": "finished"}
    response = Response(json.dumps(a, ensure_ascii=False), content_type=content_type)
    return response

@views.route('/', methods=['GET'])
def index():
    query = request.args.get('search-input', type=str)
    if not os.path.exists(csv_file_path):
        process_data()
    if not query:
        return render_template('index.html')
    start = time.time()
    if not os.path.exists(tfidf_file_path):
        get_tftidf()
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

@views.route('/get_tfidf_sample')
def get_tftidf_sample():
    # Load the tf-idf index from the JSON file
    if not os.path.exists(tfidf_sample_file_path):
        start = time.time()
        tfidf_measurements = calculate_tf_idf(csv_sample_file_path)
        end = time.time()
        print('TF-IDF calculation time: ', (end - start), ' sec(s)', file=sys.stderr)
        start = time.time()
        save_tf_idf(tfidf_measurements, tfidf_sample_file_path, tfidf_vocab_sample_file_path, tfidf_vectorizer_sample_file_path)
        end = time.time()
        print('TF-IDF files saving calculation time: ', (end - start), ' sec(s)', file=sys.stderr)
    
    a = {"status": "finished"}
    response = Response(json.dumps(a, ensure_ascii=False), content_type=content_type)
    return response

# Create a new endpoint to render the HTML template with the plot
@views.route('/top_keywords_speech_plot')
def display_top_keywords_speech_plot():
    start = time.time()
    csv_file, tfidf_file, tfidf_vocab_file, _ = select_plot_files()
    with open(top_keywords_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    top_speeches = data['top_speeches']
    speech_dates, speech_top_words = extract_dates_top_words_per_speech(top_speeches)

    # Plot keyword trends for each speech using Plotly
    fig = go.Figure()
    for speech_id, top_words in speech_top_words.items():
        fig.add_trace(go.Scatter(y=list(speech_dates[speech_id]), x=top_words, mode='markers', name=f'Speech {speech_id}'))
    fig.update_layout(
        title='Top Keywords Over Time (By Speech)',
        xaxis_title='Date',
        yaxis_title='Top Keywords',
        xaxis=dict(tickangle=45),
        legend=dict(x=1.05, y=1),
    )

    # Save the plot as an HTML file
    fig.write_html(speech_plot_html_path)
    end = time.time()
    print('Top Keywords per Speech time: ', (end - start), ' sec(s)', file=sys.stderr)
    return render_template('top_keywords_speech_plot.html')

# Create a new endpoint to render the HTML template with the plot
@views.route('/top_keywords_member_plot')
def display_top_keywords_member_plot():
    start = time.time()
    if os.path.exists(member_plot_html_path):
        return render_template('top_keywords_member_plot.html')
    
    # Extract dates, top words, and TF-IDF values for each member
    csv_file, tfidf_file, tfidf_vocab_file, _ = select_plot_files()
    with open(top_keywords_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    top_speeches_per_member = data['top_speeches_per_member']
    member_dates, member_top_words = extract_dates_top_words_per_member(top_speeches_per_member)

    # Plot keyword trends for each member using Plotly
    fig = go.Figure()
    for member, dates in member_dates.items():
        fig.add_trace(go.Scatter(x=member_top_words[member], y=dates, mode='markers', name=member))
    fig.update_layout(
        title='Top Keywords Over Time (By Member)',
        xaxis_title='Top Keywords',
        yaxis_title='Date',
        xaxis=dict(tickangle=45),
        legend=dict(x=1.05, y=1),
    )

    # Save the plot as an HTML file
    fig.write_html(member_plot_html_path)
    end = time.time()
    print('Top Keywords per Member time: ', (end - start), ' sec(s)', file=sys.stderr)
    return render_template('top_keywords_member_plot.html')

# Create a new endpoint to render the HTML template with the plot
@views.route('/top_keywords_party_plot')
def display_top_keywords_party_plot():
    start = time.time()
    if os.path.exists(party_plot_html_path):
        return render_template(PARTY_PLOT_PATH)
    
    # Extract dates, top words, and TF-IDF values for each member
    csv_file, tfidf_file, tfidf_vocab_file, _ = select_plot_files()
    with open(top_keywords_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    top_speeches_per_party = data['top_speeches_political_party']
    party_dates, party_top_words = extract_dates_top_words_per_party(top_speeches_per_party)

    # Plot keyword trends for each party using Plotly
    fig = go.Figure()
    for party, dates in party_dates.items():
        fig.add_trace(go.Scatter(x=party_top_words[party], y=dates, mode='markers', name=party))
    fig.update_layout(
        title='Top Keywords Over Time (By party)',
        xaxis_title='Top Keywords',
        yaxis_title='Date',
        xaxis=dict(tickangle=45),
        legend=dict(x=1.05, y=1),
    )

    # Save the plot as an HTML file
    fig.write_html(party_plot_html_path)
    end = time.time()
    print('Top Keywords per Party time: ', (end - start), ' sec(s)', file=sys.stderr)
    return render_template('top_keywords_party_plot.html')

@views.route('/pairwise_similarities', methods = ["GET"])
def pairwise_similarities():
    csv_file, tfidf_file, tfidf_vocab_file, tfidf_vec_file = select_plot_files()
    k = int(request.args.get('kval', 10))
    start = time.time()
    top_k_pairs = get_top_k_pairwise_similarities(csv_file, tfidf_vec_file, k, SIMILARITY_THRESHOLD)
    end = time.time()
    print('Pairwise Similarities calculation time: ', (end - start), ' sec(s)', file=sys.stderr)
    k = len(top_k_pairs)
    construct_ps_graph(top_k_pairs, k)
    return render_template('pairwise_similarities.html', top_k_pairs=top_k_pairs)

@views.route('/lsa')
def lsa():
    start = time.time()
    csv_file, tfidf_file, tfidf_vocab_file, tfidf_vec_file = select_plot_files()
    topics = get_topics(tfidf_file, tfidf_vec_file)
    end = time.time()
    print('LSA calculation time: ', (end - start), ' sec(s)', file=sys.stderr)
    return render_template('lsa.html', topics=topics)

@views.route('/clustering')
def clustering():
    start = time.time()
    k = int(request.args.get('kval', 10))
    if k > CLUSTERING_LIMIT: k = CLUSTERING_LIMIT
    csv_file, tfidf_file, tfidf_vocab_file, tfidf_vec_file = select_plot_files()
    matrix_k = get_matrix_k(tfidf_file)
    clusters = get_clusters(matrix_k, csv_file, k)
    end = time.time()
    print('Clustering time: ', (end - start), ' sec(s)', file=sys.stderr)
    return render_template('clustering.html', clusters=clusters)

def select_plot_files():
    global no_data_rows, no_sample_rows
    if not os.path.exists(csv_file_path):
        no_data_rows, _ = create_files(DATA_FOLDER, INPUT_DATA_FILE, DATA_FILE, SAMPLE_DATA_FILE, FRACTION)
    print(no_data_rows, file=sys.stderr)
    csv_file, tfidf_file, tfidf_vocab_file, tfidf_vec_file = csv_file_path, tfidf_file_path, tfidf_vocab_file_path, tfidf_vectorizer_file_path
    if no_data_rows > ROWS_LIMIT:
        csv_file, tfidf_file, tfidf_vocab_file, tfidf_vec_file = csv_sample_file_path, tfidf_sample_file_path, tfidf_vocab_sample_file_path, tfidf_vectorizer_sample_file_path
        if not os.path.exists(tfidf_sample_file_path): get_tftidf_sample()
    elif no_data_rows <= ROWS_LIMIT:
        if not os.path.exists(tfidf_file_path): get_tftidf()
    if not os.path.exists(top_keywords_file_path):
        get_top_keywords(csv_file, tfidf_file, tfidf_vocab_file, top_keywords_file_path)
    return (csv_file, tfidf_file, tfidf_vocab_file, tfidf_vec_file)