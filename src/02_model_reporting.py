# -*- coding: utf-8 -*-
"""
URL Link Scam Detection using Python and Scikit-Learn

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

model_choices = {
            'log_reg': LogisticRegressionCV(max_iter=1000, ), # this will act as our baseline model
            'sgd':SGDClassifier(max_iter=20, loss='log', random_state=42),
            'dt':DecisionTreeClassifier(random_state=42, min_samples_leaf=20),
            'mlp':MLPClassifier(hidden_layer_sizes=(512, 256, 128,64, 32), activation='tanh', max_iter=100, random_state=42),
            'xgb_model':xgb.XGBClassifier(objective="binary:logistic", random_state=42), # also available is multi:softprob
            #'svc':SVC(gamma='scale', probability=True, random_state=42)
            'random_forest': RandomForestClassifier(random_state=42, min_samples_leaf=20),
            'extra_trees': ExtraTreesClassifier(random_state=42, min_samples_leaf=20)
    }

text_fetcher = FunctionTransformer(get_text_data)
other_data_fetcher = FunctionTransformer(get_all_other_data)
log_transformer = FunctionTransformer(log_transform)
at_symbol_detector = FunctionTransformer(get_at_symbol_checks)
shortening_service_detector = FunctionTransformer(get_shortening_service_checks)
url_length_categorizer = FunctionTransformer(get_url_length_categories)
url_length_calculator = FunctionTransformer(get_url_lengths)
ip_address_detector = FunctionTransformer(get_ip_address_checks)
sub_domain_detector = FunctionTransformer(get_number_of_sub_domain_checks)
prefix_suffix_detector = FunctionTransformer(get_prefix_suffix_positions)

double_slash_detector = FunctionTransformer(get_double_slash_positions)
dir_path_counter = FunctionTransformer(get_number_of_dir_paths)
digit_to_letter_counter = FunctionTransformer(get_digit_to_letter_ratio)
digits_counter = FunctionTransformer(get_number_of_digits)
www_counter = FunctionTransformer(get_number_of_www_checks)
query_term_counter = FunctionTransformer(get_number_of_query_term_checks)
special_character_counter = FunctionTransformer(get_number_of_special_character_checks)

url_field = ['domain']

df_metrics = pd.DataFrame()

for model_choice in model_choices.keys():
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
    
    # mlpipe = Pipeline([
    #   ('union', FeatureUnion(
    #     transformer_list=[
    #       ('alexa_features', Pipeline([
    #                     ('numeric_selector', other_data_fetcher),
    #                     ('log_transformer', log_transformer),
    #                     ('minmaxscaler', StandardScaler())
    #                 ])),
    #       ('url_lexical_features',
    #                     ColumnTransformer([
    #                         ('ip_address_detector', ip_address_detector, url_field),
    #                         ('at_symbol_detector', at_symbol_detector, url_field),
    #                         ('shortening_service_detector', shortening_service_detector, url_field),
    #                         ('sub_domain_detector', sub_domain_detector, url_field),
    #                         ('prefix_suffix_detector', prefix_suffix_detector, url_field),
    #                     ])
    #           ),
    #       ('url_numerical_features', Pipeline([
    #                         ('url_numeric_transformer', ColumnTransformer([
    #                             ('double_slash_detector', double_slash_detector, url_field),
    #                             ('url_length_calculator', url_length_calculator, url_field),
    #                             ('dir_path_counter', dir_path_counter, url_field),
    #                             ('digit_to_letter_counter', digit_to_letter_counter, url_field),
    #                             ('digits_counter', digits_counter, url_field),
    #                             ('www_counter', www_counter, url_field),
    #                             ('query_term_counter', query_term_counter, url_field),
    #                             ('special_character_counter', special_character_counter, url_field)
                            
    #                         ])),
    #                         ('minmaxscaler', StandardScaler())
    #           ])),
    #       ('url_text_features', Pipeline([
    #                     ('text_selector', text_fetcher),
    #                     ('hasher', HashingVectorizer(decode_error='ignore', 
    #                                n_features=2 ** 18,
    #                                alternate_sign=False,
    #                                ngram_range=(2, 3), 
    #                                analyzer="char")),
    #                     ('tfidf_transformer', TfidfTransformer()),
    #                     ('svd', TruncatedSVD(n_components=50))
    #                 ]))
          
    #     ])),
    #     ('model_selection', model_choices[model_choice])  
    # ])
    
    
    if TRAIN_MODEL_FROM_SCRATCH:
      # 2.1 fit the classifiers
      mlpipe.fit(X_train, y_train)
    
      # 2.2 save model to file
      d1 = MODEL_RUN_DATE.strftime("%d_%m_%Y")
      dump(mlpipe, MODEL_OUTPUT_PATH + '_' + model_choice + '_url_spam_detection_pipeline_' + d1 + '.joblib') 


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
    df_metrics = pd.concat([df_metrics, df_metrics_temp], axis = 0)

df_metrics.to_csv(ANALYSIS_PATH + 'basic_features_model_experimentation_metrics.csv')

