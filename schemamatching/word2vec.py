import re
import gensim
import numpy as np
import pandas as pd

import spacy

from schemamatching.instancedata import collect_instance_data

def compare_tag_names(xml1, xml2):
    def last_item(tag):
        tag = re.sub('{.*}', '', tag)
        tag = re.sub('#', '/', tag)
        return tag.split('/')[-1]
    
    def preprocess(word):
        if("_" in word):
            tokens = word.split("_")
        elif("-" in word):
            tokens = word.split("-")
        elif(True in map(lambda l: l.isupper(), word)):
            tokens = re.sub('(?!^)([A-Z][a-z]+)', r' \1', word).split()
        else :
            tokens = [word]
        tokens = [t.lower() for t in tokens]
        return ' '.join(tokens)

    nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])
    
    tags1 = collect_instance_data(xml1).keys()
    tags2 = collect_instance_data(xml2).keys()
    
    np_matrix = np.zeros((len(tags1), len(tags2)), dtype=float)
    df = pd.DataFrame(np_matrix, index=tags1, columns=tags2)
    
    for t1 in tags1:
        for t2 in tags2:
            token1 = nlp(preprocess(last_item(t1)))
            token2 = nlp(preprocess(last_item(t2)))
            try:
                df.loc[t1, t2] = token1.similarity(token2)
            except:
                print('Error: ', token1, token2)
    
    df = df.div(df.sum(axis=1), axis=0)
    return df