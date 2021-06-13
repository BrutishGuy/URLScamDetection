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

from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

from src.feature_helper_functions import *

def get_text_data(df):
  return df['domain']

def get_all_other_data(df):
  return df.drop(['domain'], axis = 1)

pd.set_option('display.max_columns', 500)

DATA_FOLDER = './data/'
MODEL_OUTPUT_PATH = './output/'
TRAIN_MODEL_FROM_SCRATCH = True
MODEL_RUN_DATE = datetime.today()

url_data = []
for line in gzip.open(DATA_FOLDER + "urlset.json.gz", 'r'):
    url_data.append(json.loads(line))
    
df_url_data = pd.DataFrame(url_data)


sns.histplot(x=np.log(df_url_data['ranking']))

# Feature Engineering
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

mlpipe = Pipeline([
      ('union', FeatureUnion(
        transformer_list=[
          ('alexa_features', Pipeline([
                        ('numeric_selector', other_data_fetcher),
                        ('log_transformer', log_transformer),
                        ('minmaxscaler', StandardScaler())
                    ])),
          ('url_lexical_features',
                        ColumnTransformer([
                            ('ip_address_detector', ip_address_detector, url_field),
                            ('at_symbol_detector', at_symbol_detector, url_field),
                            ('shortening_service_detector', shortening_service_detector, url_field),
                            ('sub_domain_detector', sub_domain_detector, url_field),
                            ('prefix_suffix_detector', prefix_suffix_detector, url_field),
                        ])
              ),
          ('url_numerical_features', Pipeline([
                            ('url_numeric_transformer', ColumnTransformer([
                                ('double_slash_detector', double_slash_detector, url_field),
                                ('url_length_calculator', url_length_calculator, url_field),
                                ('dir_path_counter', dir_path_counter, url_field),
                                ('digit_to_letter_counter', digit_to_letter_counter, url_field),
                                ('digits_counter', digits_counter, url_field),
                                ('www_counter', www_counter, url_field),
                                ('query_term_counter', query_term_counter, url_field),
                                ('special_character_counter', special_character_counter, url_field)
                            
                            ])),
                            ('minmaxscaler', StandardScaler())
              ])),
          ('url_text_features', Pipeline([
                        ('text_selector', text_fetcher),
                        ('hasher', HashingVectorizer(decode_error='ignore', 
                                   n_features=2 ** 18,
                                   alternate_sign=False,
                                   ngram_range=(2, 3), 
                                   analyzer="char")),
                        ('tfidf_transformer', TfidfTransformer()),
                        ('svd', TruncatedSVD(n_components=50))
                    ]))
          
        ])),
    ])

df_url_data_features = feature_pipe.fit_transform(df_url_data.drop(['label'], axis = 1))
df_url_data_features['label'] = df_url_data['label']

# Exploratory Analytics on Engineered Features

sns.histplot(x=np.log(df_url_data['url_length']))

