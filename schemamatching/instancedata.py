import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def create_pipeline(classifier):            
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2), lowercase=False)
    pipeline = make_pipeline(vectorizer, classifier)
    return pipeline

def collect_instance_data(xml_string):
    def _recurse(data, current_path, xml_element):
        current_path = current_path + '/' + xml_element.tag
        for key in xml_element.attrib:
            attr_path = current_path + '#' + key
            if attr_path not in data:
                data[attr_path] = []
            data[attr_path].append(xml_element.attrib[key])
        if len(xml_element) == 0:
            if current_path not in data:
                data[current_path] = []
            data[current_path].append(xml_element.text)
        else:    
            for child in xml_element:
                _recurse(data, current_path, child)
    data = {}
    xml_tree = ET.fromstring(xml_string)
    _recurse(data, '', xml_tree)
    return data

def get_tag_names(xml_string):
    return collect_instance_data(xml_string).keys()

def get_features(xml):
    records = []
    instance_data_dict = collect_instance_data(xml)
    for tag in instance_data_dict:
        for instance in instance_data_dict[tag]:
            records.append((instance, tag))
    return pd.DataFrame.from_records(records, columns=['item', 'tag'])

def compare_xmls(xml1, xml2, model=None):
    xml1_data = collect_instance_data(xml1)
    xml2_data = collect_instance_data(xml2)
    xml2_features = get_features(xml2)

    if model == None: model = DecisionTreeClassifier(random_state=42)
    pipeline = create_pipeline(model)
    pipeline.fit(xml2_features['item'], xml2_features['tag'])

    output_shape =len(xml1_data.keys()), len(xml2_data.keys())
    outputs = pd.DataFrame(np.zeros(output_shape),
        index=xml1_data.keys(), columns=xml2_data.keys())
    
    for tag in xml1_data:
        predictions = pipeline.predict(xml1_data[tag])
        total = len(predictions)
        for p in predictions:
            outputs.loc[tag, p] += 1.0 / total
    return outputs
