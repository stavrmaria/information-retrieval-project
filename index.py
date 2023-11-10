import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import string
import json
from snowballstemmer import stemmer

def process_text(text):
    greek_stemmer = stemmer("greek")
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)

    print(text, file=sys.stderr)

    stop = set(stopwords.words('greek') + list(string.punctuation))
    tokens = [i for i in word_tokenize(text.lower()) if i not in stop]
    stemmed_tokens = [greek_stemmer.stemWord(token) for token in tokens]
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
                if doc_id in inverted_index[term][1]:
                    # Increase n_t of the term and update the (doc_id, freq) entry
                    inverted_index[term][1][doc_id] += 1
                else:
                    inverted_index[term][1][doc_id] = 1
                    inverted_index[term][0] += 1
            else:
                entries = {doc_id: 1}
                inverted_index[term] = [1, entries]
    
    return inverted_index