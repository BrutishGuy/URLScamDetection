# -*- coding: utf-8 -*-
"""
@author: VictorGueorguiev
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gzip
import json  
import seaborn as sns
import urllib.parse as urlparse
from datetime import datetime

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb

from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from joblib import dump, load

from src.feature_helper_functions import *

from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from hyperopt import tpe


def get_text_data(df):
    df['domain'] = df['domain'].astype(str)
    return df['domain']

def get_all_other_data(df):
    return df.drop(['domain'], axis = 1)

pd.set_option('display.max_columns', 500)
pd.options.mode.chained_assignment = None

DATA_FOLDER = './data/'
MODEL_OUTPUT_PATH = './output/'
TRAIN_MODEL_FROM_SCRATCH = True
MODEL_RUN_DATE = datetime.today()
ANALYSIS_PATH = './analysis/'



url_data = []
for line in gzip.open(DATA_FOLDER + "urlset.json.gz", 'r'):
    url_data.append(json.loads(line))
    
df_url_data = pd.DataFrame(url_data)

X = df_url_data[['domain', 'ranking']]
y = df_url_data.label

# NB the dataset is already very balanced, so we do not take additional measures to balance the data

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.25, random_state=42)

text_fetcher = FunctionTransformer(get_text_data)
other_data_fetcher = FunctionTransformer(get_all_other_data)
log_transformer = FunctionTransformer(log_transform)

mlpipe = Pipeline([
  ('text_selector', text_fetcher),
  ('hasher', HashingVectorizer(decode_error='ignore', 
                                n_features=2 ** 18,
                                alternate_sign=False,
                                ngram_range=(2, 3), 
                                analyzer="char")),
  ('tfidf_transformer', TfidfTransformer()),
  ('svd', TruncatedSVD(n_components=60)),
  ('model_selection', HyperoptEstimator(classifier=any_classifier('my_clf'),
                      preprocessing=[],
                      algo=tpe.suggest,
                      max_evals=100,
                      trial_timeout=120))
])

mlpipe.fit(X_train, y_train)

y_pred = mlpipe.predict(X_test)
y_pred_proba = mlpipe.predict_proba(X_test)[:, 1]
df_metrics_temp = pd.DataFrame(columns=['RUN_DATE', 'MODEL', 'ACCURACY', 'F1 SCORE MICRO AVG', 'PRECISION SCORE MICRO AVG', 'RECALL SCORE MICRO AVG', 'ROCAUC MICRO AVG'])
df_metrics_temp.loc[0] = [MODEL_RUN_DATE, 
                          model_choice,
                            accuracy_score(y_pred, y_test),
                            f1_score(y_pred, y_test),
                            precision_score(y_pred, y_test),
                            recall_score(y_pred, y_test),
                            roc_auc_score(y_score=y_pred_proba, y_true=y_test)]

print(df_metrics_temp)


