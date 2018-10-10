import numpy as np
import pandas as pd
from functools import reduce

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer, MinMaxScaler, LabelEncoder, PolynomialFeatures, OneHotEncoder, LabelBinarizer, FunctionTransformer, StandardScaler

class DFFeatureUnion(TransformerMixin, BaseEstimator):
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list
    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self
    def transform(self, X):
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        assert X.shape[0] == Xunion.shape[0]
        return Xunion

class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.cols]

class DummyTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.dv = None
    def fit(self, X, y=None):
        Xdict = X.to_dict('records')
        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(Xdict)
        return self
    def transform(self, X):
        Xdict = X.to_dict('records')
        Xt = self.dv.transform(Xdict)
        cols = self.dv.get_feature_names()
        Xdum = pd.DataFrame(Xt, index=X.index, columns=cols)
        nan_cols = [c for c in cols if '=' not in c]
        Xdum = Xdum.drop(nan_cols, axis=1)
        return Xdum

class DFTextVectorizers(TransformerMixin, BaseEstimator):
    def __init__(self,  Vectorizer=CountVectorizer, *args, **kwargs):
        self.Vectorizer = Vectorizer
        self.args = args
        self.kwargs = kwargs
        self.vectorizers = {}
    def fit(self, X, y=None):
        for column in X.columns:
            self.vectorizers[column] = self.Vectorizer(*self.args, **self.kwargs)
            self.vectorizers[column].fit(X[column])
        return self
    def _col_transform(self, X, col):
        Xvec = self.vectorizers[col].transform(X[col])
        columns = self.vectorizers[col].get_feature_names()
        columns = list(map(lambda c: col + '_token=' + c, columns))
        return pd.SparseDataFrame(Xvec, index=X.index, columns=columns)
    def transform(self, X):
        Xvecs = [self._col_transform(X, column) for column in X.columns]
        return reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xvecs)

class MultiEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, sep=','):
        self.sep = sep
        self.mlbs = None
    def fit(self, X, y=None):
        Xsplit = X.applymap(lambda x: x.split(self.sep))
        self.mlbs = [MultiLabelBinarizer().fit(Xsplit[c]) for c in X.columns]
        return self
    def _col_transform(self, x, mlb):
        cols = [''.join([x.name, '=', c]) for c in mlb.classes_]
        xmlb = mlb.transform(x)
        xdf = pd.DataFrame(xmlb, index=x.index, columns=cols)
    def transform(self, X):
        Xsplit = X.applymap(lambda x: x.aplit(self.sep))
        Xmlbs = [self._col_transform(Xsplit[c], self.mlbs[i]) for i, c in enumerate(X.columns)]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xmlbs)
        return Xunion

class DFScaler(TransformerMixin):
    def __init__(self, Scaler=StandardScaler, *args, **kwargs):
        self.Scaler = Scaler
        self.args = args
        self.kwargs = kwargs
        self.scaler = None
    def fit(self, X, y=None):
        self.scaler = self.Scaler(*self.args, **self.kwargs)
        self.scaler.fit(X)
        return self
    def transform(self, X):
        Xt = self.scaler.transform(X)
        return pd.DataFrame(Xt, index=X.index, columns=X.columns)
class DFImputer(TransformerMixin, BaseEstimator):
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imp = None
        self.statistics_ = None
    def fit(self, X, y=None):
        self.imp = Imputer(strategy=self.strategy)
        self.imp.fit(X)
        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
    def transform(self, X):
        Ximp = self.imp.transform(X)
        return pd.DataFrame(Ximp, index=X.index, columns=X.columns)
        
class DFStandardScaler(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None
    def fit(self, X, y=None):
        self.ss = StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self
    def transform(self, X):
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled

class DateFormatter(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.apply(pd.to_datetime)
