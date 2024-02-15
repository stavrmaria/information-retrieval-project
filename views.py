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
from pairwise_similarities import get_top_k_pairwise_similarities
from lsa import get_topics
from clustering import get_matrix_k, get_clusters

DATA_FILE = 'Greek_Parliament_Proceedings_1989_2020.csv'
# DATA_FILE = 'Greek_Parliament_Proceedings_1989_2020_sample.csv'
# DATA_FILE = 'sample_data.csv'
DATA_FILE = 'output_file.csv'
SAMPLE_DATA_FILE = 'output_sample.csv'
# DATA_FILE = 'test_data.csv'

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

content_type='application/json; charset=utf-8'
views = Blueprint(__name__, "views")

@views.route('/', methods=['GET'])
def index():
    query = request.args.get('search-input', type=str)
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
@views.route('/top_keywords_member_plot')
def display_top_keywords_member_plot():
    if os.path.exists(member_plot_html_path):
        return render_template('top_keywords_member_plot.html')
    
    # Extract dates, top words, and TF-IDF values for each member
    if not os.path.exists(top_keywords_file_path):
        get_top_keywords(csv_sample_file_path, tfidf_sample_file_path, tfidf_vocab_sample_file_path, top_keywords_file_path)
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
    plot_html_path = 'templates/top_keywords_member_plot.html'
    fig.write_html(plot_html_path)

    # Render the HTML template with the plot
    return render_template('top_keywords_member_plot.html')

# Create a new endpoint to render the HTML template with the plot
@views.route('/top_keywords_party_plot')
def display_top_keywords_party_plot():
    if os.path.exists(party_plot_html_path):
        return render_template('top_keywords_party_plot.html')

    # Extract dates, top words, and TF-IDF values for each party
    if not os.path.exists(top_keywords_file_path):
        get_top_keywords(csv_sample_file_path, tfidf_sample_file_path, tfidf_vocab_sample_file_path, top_keywords_file_path)
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
    plot_html_path = 'templates/top_keywords_party_plot.html'
    fig.write_html(plot_html_path)

    # Render the HTML template with the plot
    return render_template('top_keywords_party_plot.html')

# Create a new endpoint to render the HTML template with the plot
@views.route('/top_keywords_speech_plot')
def display_top_keywords_speech_plot():
    # Extract dates, top words, and TF-IDF values for each speech
    if not os.path.exists(top_keywords_file_path):
        get_top_keywords(csv_sample_file_path, tfidf_sample_file_path, tfidf_vocab_sample_file_path, top_keywords_file_path)
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
    plot_html_path = 'templates/top_keywords_speech_plot.html'
    fig.write_html(plot_html_path)

    # Render the HTML template with the plot
    return render_template('top_keywords_speech_plot.html')

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

@views.route('/pairwise_similarities')
def pairwise_similarities():
    k = 100
    start = time.time()
    top_k_pairs = get_top_k_pairwise_similarities(csv_sample_file_path, tfidf_vectorizer_file_path, k, SIMILARITY_THRESHOLD)
    end = time.time()
    print('Pairwise Similarities calculation time: ', (end - start), ' sec(s)', file=sys.stderr)

    graph = nx.Graph()
    # Add edges with weights from top k pairs
    for pair in top_k_pairs:
        name1, name2, weight = pair
        graph.add_edge(name1, name2, weight=weight)
    
    # Extract node positions for plotting
    pos = nx.spring_layout(graph)

    # Create edges
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    # Create nodes
    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # Add node info to hover text
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{adjacencies[0]}<br># of connections: {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f'Top {k} Pairs Graph',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    # Show plot
    fig.show()

    return render_template('pairwise_similarities.html', top_k_pairs=top_k_pairs)

@views.route('/lsa')
def lsa():
    start = time.time()
    topics = get_topics(tfidf_sample_file_path, tfidf_vectorizer_sample_file_path)
    end = time.time()
    print('LSA calculation time: ', (end - start), ' sec(s)', file=sys.stderr)
    response = Response(json.dumps(topics, ensure_ascii=False), content_type=content_type)
    return response

@views.route('/clustering')
def clusterin():
    start = time.time()
    matrix_k = get_matrix_k(tfidf_sample_file_path)
    clusters = get_clusters(matrix_k, csv_sample_file_path)
    end = time.time()
    print('Clustering time: ', (end - start), ' sec(s)', file=sys.stderr)
    response = Response(json.dumps(clusters, ensure_ascii=False), content_type=content_type)
    return response