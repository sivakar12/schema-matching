from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

from .pipeline_components import DFFeatureUnion, DummyTransformer, ColumnExtractor
from .metrics import *

class SchemaMatcher:
    def __init__(self, xml1, xml2, true_pairs, classifier=LogisticRegression(), \
            feature_selector=None, length=False, datatype=False):
        self.xml1 = xml1
        self.xml2 = xml2
        self.true_mappings_dict = make_true_mappings_dict(true_pairs)
        self.true_mappings_matrix = make_true_mappings_dataframe(true_pairs)
        self.pipeline = create_pipeline(classifier, feature_selector, length, datatype)
        self.results = None

    def generate_mappings(self):
        xml1_data = collect_instance_data(self.xml1)
        xml2_data = collect_instance_data(self.xml2)
        xml2_features = get_features(self.xml2)

        self.pipeline.fit(xml2_features['content'], xml2_features['tag'])
        results_shape = len(xml1_data.keys()), len(xml2_data.keys())
        self.results = pd.DataFrame(np.zeros(results_shape),
            index=xml1_data.keys(), columns=xml2_data.keys())
        
        for tag in xml1_data:
            X_new = pd.DataFrame({ 'content': xml1_data[tag] })
            predictions = self.pipeline.predict(X_new['content'])
            total = len(predictions)
            for p in predictions:
                self.results.loc[tag, p] += 1.0 / total
        
        return self.results
    
    def get_all_scores(self):
        functions = [accuracy, precision, recall, \
            mean_difference, average_log_loss]
        return { f.__name__: f(self.true_mappings_matrix, self.results)
            for f in functions }
    
    def get_score(self, scoring_function):
        return scoring_function(self.true_mappings_matrix, self.results)
    
    def do_internal_comparison(self):
        xml1_feature_matrix = get_features_for_specific_tags(self.xml1, self.true_mappings_dict.keys())
        predictions = self.pipeline.predict(xml1_feature_matrix['content'])
        true = list(map(lambda x: self.true_mappings_dict[x], 
            xml1_feature_matrix['tag'].values))
        return true, predictions
    
    def score_internally(self):
        true, predictions = self.do_internal_comparison()
        return {
            'accuracy': accuracy_score(true, predictions),
            'precision': precision_score(true, predictions, average='weighted'),
            'recall': recall_score(true, predictions, average='weighted')
        }


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
    return pd.DataFrame.from_records(records, columns=['content', 'tag'])

def get_features_for_specific_tags(xml, tags):
    records = []
    instance_data_dict = collect_instance_data(xml)
    for tag in instance_data_dict:
        if tag not in tags:
            continue
        for instance in instance_data_dict[tag]:
            records.append((instance, tag))
    return pd.DataFrame.from_records(records, columns=['content', 'tag'])

def datatype(a):
    try:
        float(a)
        if("." in a):
            return "float"
        else:
            return "int"
    except:
        return "string"

class LengthTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_new = pd.DataFrame()
        X_new['content'] = X.str.len()
        return X_new
        # return X['content'].str.len()

class DataTypeTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_new = pd.DataFrame()
        X_new['content'] = X.apply(datatype)
        return X_new
        # return X['content'].apply(datatype)

def create_pipeline(classifier=LogisticRegression(), feature_selector=None,\
        length=False, datatype=False):
    pipeline_items = []
    feature_extractors = []

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), lowercase=False)
    feature_extractors.append(('vectorizer', vectorizer))
    if length:
        feature_extractors.append(('length', LengthTransformer()))
    if datatype:
        feature_extractors.append(('datatype', Pipeline([
            ('apply', DataTypeTransformer()),
            ('dummify', DummyTransformer())
        ])))
    features = FeatureUnion(feature_extractors)

    pipeline_items.append(('features', features))
    if feature_selector:
        pipeline_items.append(('feature_selector', feature_selector))
    pipeline_items.append(('classifier', classifier))
    
    pipeline = Pipeline(pipeline_items)
    return pipeline

def score_pipeline(xml, pipeline, scoring_methods):
    _average = lambda numbers: sum(numbers) / len(numbers)
    
    features = get_features(xml)
    x = features['content']
    y = features['tag']
    results = { scoring_method: _average(cross_val_score(pipeline, x, y, scoring=scoring_method)) for scoring_method in scoring_methods }
    return results

def compare_xmls(xml1, xml2, model=None, \
        feature_selector=None, length=False, datatype=False):
    xml1_data = collect_instance_data(xml1)
    xml2_data = collect_instance_data(xml2)
    xml2_features = get_features(xml2)

    if model == None: model = DecisionTreeClassifier(random_state=42)
    pipeline = create_pipeline(model, feature_selector, length, datatype)
    pipeline.fit(xml2_features['content'], xml2_features['tag'])

    output_shape =len(xml1_data.keys()), len(xml2_data.keys())
    outputs = pd.DataFrame(np.zeros(output_shape),
        index=xml1_data.keys(), columns=xml2_data.keys())
    
    for tag in xml1_data:
        X_new = pd.DataFrame({ 'content': xml1_data[tag] })
        predictions = pipeline.predict(X_new['content'])
        total = len(predictions)
        for p in predictions:
            outputs.loc[tag, p] += 1.0 / total
    return outputs
