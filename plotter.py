import heapq
import os
import json
import pickle
from flask import redirect
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

from scipy import sparse

def update_heap(heap, new_item, no_keywords=10):
    if len(heap) < no_keywords:
        heapq.heappush(heap, new_item)
    elif new_item > heap[0]:
        heapq.heapreplace(heap, new_item)
    
def extract_top_keywords(heaps):
    for property_name, heap in heaps.items():
        top_keywords_per_property = [(tfidf, word, date) for (tfidf, word, date) in sorted(heap, reverse=True)]
        heaps[property_name] = top_keywords_per_property

def get_top_keywords(csv_file_path, tfidf_file_path, tfidf_vocab_file_path, top_keywords_file_path, no_keywords=10):
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
        member_names = df_chunk[1].tolist()
        political_parties = df_chunk[0].tolist()
        dates = df_chunk[2].tolist()

        no_speeches = len(speeches) - 1

        # Iterate over each speech in the chunk
        for doc_idx in range(no_speeches):
            # Extract TF-IDF values for the current speech and get the indices of the top keywords based on TF-IDF values
            tfidf_values = tfidf_matrix[doc_idx].toarray()[0]
            top_indices = tfidf_values.argsort()[-np.count_nonzero(tfidf_values):][::-1] if np.count_nonzero(tfidf_values) < no_keywords else tfidf_values.argsort()[-no_keywords:][::-1]
            top_words = [{"word": feature_names[idx], "tfidf": tfidf_values[idx], "date": dates[doc_idx + 1]} for idx in top_indices]
            
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
                current_item = (tfidf_values[top_id], feature_names[top_id], dates[doc_idx + 1])
                update_heap(top_per_member[member_name], current_item)
                update_heap(top_per_political_party[political_party], current_item)
            
    # Extract the top keywords from the max heap for each member and political party
    extract_top_keywords(top_per_member)
    extract_top_keywords(top_per_political_party)

    # Create a JSON response with the top speeches and keywords
    json_object = json.dumps(top_keywords, ensure_ascii=False, indent=4)
    with open(top_keywords_file_path, 'w', encoding='utf-8') as response:
        response.write(json_object)
    return redirect('/top_keywords_plot')

# Function to extract dates, top words, and TF-IDF values for each member
def extract_dates_top_words_per_member(data):
    member_dates = {}
    member_top_words = {}
    
    for member, speech_data in data.items():
        member_dates[member] = []
        member_top_words[member] = []
        
        for item in speech_data:
            date = datetime.strptime(item[2], '%d/%m/%Y')
            word = item[1]
            member_dates[member].append(date)
            member_top_words[member].append(word)
    
    return member_dates, member_top_words

# Function to extract dates, top words, and TF-IDF values for each political party
def extract_dates_top_words_per_party(data):
    party_dates = {}
    party_top_words = {}
    
    for party, speech_data in data.items():
        party_dates[party] = []
        party_top_words[party] = []
        
        for item in speech_data:
            date = datetime.strptime(item[2], '%d/%m/%Y')
            word = item[1]
            party_dates[party].append(date)
            party_top_words[party].append(word)
    
    return party_dates, party_top_words
