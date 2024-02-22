import itertools
import multiprocessing
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
import plotly.graph_objects as go

def calculate_similarity(pair, member_feature_vectors, threshold):
    member1, member2 = pair
    if member1 > member2:
        return None
    similarity_score = cosine_similarity(np.asarray(member_feature_vectors[member1]), np.asarray(member_feature_vectors[member2]))[0][0]
    if similarity_score >= threshold:
        return (member1, member2, similarity_score)
    else:
        return None

def calculate_similarity_wrapper(args):
    return calculate_similarity(*args)

def process_chunk(chunk, tfidf_vectorizer):
    member_feature_vectors = {}
    # Extract unique members in the current chunk
    for member, group in chunk.groupby('member_name'):
        member_tfidf_values = tfidf_vectorizer.transform(group['speech'])
        member_feature_vector = member_tfidf_values.mean(axis=0)
        member_feature_vectors[member] = member_feature_vector
    return member_feature_vectors

def process_chunk_wrapper(args):
    return process_chunk(*args)

def get_top_k_pairwise_similarities(csv_file_path, tfidf_vectorizer_file_path, k, threshold):
    # Load the TF-IDF matrix and vectorizer from the files
    with open(tfidf_vectorizer_file_path, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)

    # Load the speeches data from the data file
    chunksize = 100000
    speeches_data = pd.read_csv(csv_file_path, usecols=['member_name', 'speech'], chunksize=chunksize)
    
    # Contruct the member Feature Vector
    member_feature_vectors = {}
    with ProcessPoolExecutor() as executor:
        chunk_tfidf_vectorizer = [(chunk, tfidf_vectorizer) for chunk in speeches_data]
        results = executor.map(process_chunk_wrapper, chunk_tfidf_vectorizer)
    
    for result in results:
        for member, vector in result.items():
            if member in member_feature_vectors:
                member_feature_vectors[member] += vector
            else:
                member_feature_vectors[member] = vector
    
    members = list(member_feature_vectors.keys())
    
    with multiprocessing.Pool() as pool:
        similarity_scores = pool.starmap(calculate_similarity, 
                                         ((pair, member_feature_vectors, threshold) for pair in itertools.permutations(members, 2)))
    similar_pairs = [result for result in similarity_scores if result is not None]

    top_k_pairs = sorted(similar_pairs, key=lambda x: x[2], reverse=True)[:k]

    return top_k_pairs

def construct_ps_graph(top_k_pairs, k):
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