import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET
from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from .pipeline_components import DFFeatureUnion, DummyTransformer, ColumnExtractor

def datatype(a):
    try:
        float(a)
        if("." in a):
            # return "float"
            return 0
        else:
            # return "int"
            return 1
    except:
        # return "string"
        return 2

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

def create_pipeline(classifier, feature_selection=False, length=False, datatype=False):            
    pipeline_items = []
    feature_extractors = []

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2), lowercase=False)
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
    if feature_selection:
        pipeline_items.append(('feature_selection', SelectFromModel(classifier)))
    pipeline_items.append(('classifier', classifier))
    
    pipeline = Pipeline(pipeline_items)
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
    return pd.DataFrame.from_records(records, columns=['content', 'tag'])

def compare_xmls(xml1, xml2, model=None, \
        feature_selection=False, length=False, datatype=False):
    xml1_data = collect_instance_data(xml1)
    xml2_data = collect_instance_data(xml2)
    xml2_features = get_features(xml2)

    if model == None: model = DecisionTreeClassifier(random_state=42)
    pipeline = create_pipeline(model, feature_selection, length, datatype)
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
