from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spacy

import re

from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

from .pipeline_components import DFFeatureUnion, DummyTransformer, ColumnExtractor
from .graph import SimilarityFlooding
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
        self.word_embedding_results = None
        self.results_after_similarity_flooding = None

    def generate_mappings(self):
        # This is here for backward compatibility after renaming method
        return self.get_classifier_results()

    def get_classifier_results(self, xml1=None, xml2=None):
        if xml1 == None: xml1 = self.xml1
        if xml2 == None: xml2 = self.xml2
        
        xml1_data = collect_instance_data(xml1)
        xml2_data = collect_instance_data(xml2)
        xml2_features = get_features(xml2)

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

    def get_word2vec_results(self):
        return compare_tag_names(self.xml1, self.xml2)
    
    def plot_joining_with_lambda(self, l_values=np.arange(0, 1, 0.01)):
        def error_after_joining(result1, result2, l, truth):
            combination = l * result1 + (l - 1) * result2
            return -mean_difference(combination, truth)
        result1 = self.get_classifier_results()
        result2 = self.get_word2vec_results()

        results = {l: error_after_joining(result1, result2, l, self.true_mappings_matrix) for l in l_values}
        plt.scatter(results.keys(), results.values())
        plt.xlabel('lambda')
        plt.ylabel('mean difference')
        return results

    def get_all_scores(self):
        if self.results == None:
            self.get_classifier_results()

        functions = [accuracy, precision, recall, f1, \
            mean_difference]
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
            'precision': precision_score(true, predictions, average='micro'),
            'recall': recall_score(true, predictions, average='micro'),
            'f1_score': f1_score(true, predictions, average='micro')
        }
    
    def get_same_xml_accuracies(self):

        def _evaluate_results(pred_matrix):
            true_matrix = np.identity(pred_matrix.shape[0])
            true_matrix = pd.DataFrame(true_matrix, index=pred_matrix.index, columns=pred_matrix.columns)
            return accuracy(true_matrix, pred_matrix)

        result_1 = self.get_classifier_results(self.xml1, self.xml1)
        result_2 = self.get_classifier_results(self.xml2, self.xml2)
        return list(map(_evaluate_results, [result_1, result_2]))
    
    def do_similarity_flooding(self):
        if self.word_embedding_results == None:
            self.word_embedding_results = compare_tag_names(
                self.xml1, self.xml2, include_parents=True)
        
        if self.results == None:
            self.get_classifier_results()
        
        sf = SimilarityFlooding(self.xml1, self.xml2)
        sf.set_initial_scores(self.word_embedding_results)
        sf.set_initial_scores(self.results)
        
        sf.flood_until_threshold()
        self.results_after_similarity_flooding = sf.get_results()
        return self.results_after_similarity_flooding


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

def get_parent_tags(tag):
    if tag.count('/') < 1:
        return []
    parent = re.match(r'(.*)/.*', tag)[1]
    return [tag] + get_parent_tags(parent)

def compare_tag_names(xml1, xml2, include_parents=False):
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
    
    if include_parents:
        tags1 = [parent_tag for tag in tags1 for parent_tag in get_parent_tags(tag)]
        tags2 = [parent_tag for tag in tags2 for parent_tag in get_parent_tags(tag)]
        
        tags1 = list(set(tags1))
        tags2 = list(set(tags2))
    
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
