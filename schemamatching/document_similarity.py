import numpy as np
import pandas as pd

from schemamatching.instancedata import *

import spacy
nlp = spacy.load('en_core_web_md', disable=['ner', 'parser'])
nlp.max_length = 5000000

def get_documents_from_xml(xml_string, separator=' '):
    instance_data = collect_instance_data(xml_string)
    return {key:separator.join(value) for key, value in instance_data.items()}

def compare_xmls_using_document_similarity(xml1, xml2):
    dict1 = get_documents_from_xml(xml1)
    dict2 = get_documents_from_xml(xml2)
    np_matrix = np.zeros((len(dict1), len(dict2)), dtype=float)
    df = pd.DataFrame(np_matrix, index=dict1.keys(), columns=dict2.keys())
    
    for tag1 in dict1.keys():
        text1 = nlp(dict1[tag1])
        for tag2 in dict2.keys():
            text2 = nlp(dict2[tag2])
            df.loc[tag1, tag2] = text1.similarity(text2)

    df = df.div(df.sum(axis=1), axis=0)
    return df