# -*- coding: utf-8 -*-
"""

@author: VictorGueorguiev
"""

# ----------------------------------------------------------------------------
# STEP 0: INITIALIZATION
# ----------------------------------------------------------------------------


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

pd.set_option('display.max_columns', 500)
pd.options.mode.chained_assignment = None

# ----------------------------------------------------------------------------
# STEP 1: ENVIRONMENT CONSTANTS AND DATA LOADING
# ----------------------------------------------------------------------------

DATA_FOLDER = './data/'
MODEL_OUTPUT_PATH = './output/'
TRAIN_MODEL_FROM_SCRATCH = True
MODEL_RUN_DATE = datetime.today()

url_data = []
for line in gzip.open(DATA_FOLDER + "urlset.json.gz", 'r'):
    url_data.append(json.loads(line))
    
df_url_data = pd.DataFrame(url_data)
# check if the dataset is balanced on the dependent variable!
df_url_data[df_url_data.label == 1].shape[0]/df_url_data.shape[0]

# ----------------------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING
# ----------------------------------------------------------------------------

# here we use helper functions defined in ./src/feature_helper_functions.py
# to define FunctionTransformers to be used with Scikit-Learn's pipeline
# model building tools to make easily transferrable, replicable and deployable models.

# see pydocs documentation in ./src/feature_helper_functions.py for descriptions
# of each features and how each transformer works
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

# for the below variables we do not do any scaling for the reason that we wish 
# to directly see behaviour against the dependent variable later graphically,
# rather than adhere to modelling best practices and model assumptions

hashing_feature_pipe = Pipeline([
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
    ])

lexical_feature_pipe = Pipeline([
      ('union', FeatureUnion(
        transformer_list=[
          ('alexa_features', Pipeline([
                        ('numeric_selector', other_data_fetcher),
                        ('log_transformer', log_transformer)
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
                            
                            ]))
              ]))
          
        ])),
    ])

# define the lexical features, later adding back the dependent variable
df_url_data_lexical_features = lexical_feature_pipe.fit_transform(df_url_data.drop(['label'], axis = 1))
df_url_data_lexical_features['label'] = df_url_data['label']

# define the hashing features, later adding back the dependent variable
df_url_data_hashing_features = hashing_feature_pipe.fit_transform(df_url_data.drop(['label'], axis = 1))
df_url_data_hashing_features['label'] = df_url_data['label']

# ----------------------------------------------------------------------------
# STEP 3: EXPLORATORY ANALYTICS ON ENGINEERED FEATURES
# ----------------------------------------------------------------------------

# checking how the ranking distributes
# using a histogram on the ranking variable
sns.histplot(x=np.log(df_url_data['ranking']))

# checking how the length of the url distributes
# using a historgram on the url_length
sns.histplot(x=np.log(df_url_data['url_length']))

# ----------------------------------------------------------------------------
# STEP 4: EXAMINING MULTI-COLLINEARITY OF FEATURES
# ----------------------------------------------------------------------------


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X).correlation
corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(
    corr_linkage, labels=data.feature_names.tolist(), ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro['ivl']))

ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
ax2.set_yticklabels(dendro['ivl'])
fig.tight_layout()
plt.show()

