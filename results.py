import pandas as pd

def get_results(query):
    results = [{"speech_start"  : "Θα ήθελα να σας μιλήσω για...",
                "name"          : "γεωργιος ιωαννου",
                "political_party" : "πασοκ",
                "date"          : "21/07/2020",
                "score"         : 0.088,
                "doc_id"         : 1},
               {"speech_start"  : "Ευχαριστώ πολύ, αλλά...",
                "name"          : "ιωαννης πρεκας",
                "political_party" : "νεα δημοκρατια",
                "date"          : "10/08/2021",
                "score"         : 0.076,
                "doc_id"         : 2}]
    return results

def get_result(doc_id, data_file_path):
    result = []

    df_speeches = pd.read_csv(data_file_path)

    doc_id = int(doc_id)
    result.append(df_speeches.values[doc_id][0]) # name
    result.append(df_speeches.values[doc_id][1]) # date
    result.append(df_speeches.values[doc_id][5]) # political party
    result.append(df_speeches.values[doc_id][10]) # speech

    return result