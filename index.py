import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import string
import json

def process_text(text):
    stop = set(stopwords.words('greek') + list(string.punctuation))
    stemmed_tokens = [i for i in word_tokenize(text.lower()) if i not in stop]
    return stemmed_tokens

# Save the inverted index to the JSON file
def save_index(inverted_index, json_file_path):
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(inverted_index, json_file, ensure_ascii=False)

def construct_inverted_index(data_file_path):
    inverted_index = {}
    nltk.download('punkt')
    nltk.download('stopwords')

    # Read the speech column of the csv file
    df_speeches = pd.read_csv(data_file_path, usecols = ['speech'])

    # Get the speech rows of the csv file
    for doc_id, row in df_speeches.iterrows():
        # Tokenize and preprocess the speech
        speech = row['speech']
        tokens = process_text(speech)

        # Construct the index based on the tokens of each speech
        for term in tokens:
            if term in inverted_index:
                inverted_index[term].append(doc_id)
            else:
                inverted_index[term] = [doc_id]
    
    return inverted_index