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
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import association_metrics as am

from feature_helper_functions import *

pd.set_option('display.max_columns', 500)
pd.options.mode.chained_assignment = None
from sklearn import set_config
set_config(display='diagram')

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

# for the below, to make plots more visibly interpretable, we use a smaller reduced dimension of 10 for the Truncated SVD.
# in the models, we capture more information and find a tuned parameter of 50 to be more optimal
hashing_feature_pipe = Pipeline([
    ('url_text_features', Pipeline([
                        ('text_selector', text_fetcher),
                        ('hasher', HashingVectorizer(decode_error='ignore', 
                                   n_features=2 ** 18,
                                   alternate_sign=False,
                                   ngram_range=(2, 3), 
                                   analyzer="char")),
                        ('tfidf_transformer', TfidfTransformer()),
                        ('svd', TruncatedSVD(n_components=10))
                    ]))
    ])

lexical_feature_pipe = Pipeline([
      ('union', FeatureUnion(
        transformer_list=[
          ('alexa_features', Pipeline([
                        ('numeric_selector', other_data_fetcher),
                        ('log_transformer', log_transformer),
                        ('minmaxscaler', MinMaxScaler())
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
df_url_data_lexical_features = pd.DataFrame(df_url_data_lexical_features)
df_url_data_lexical_features.columns = ['Alexa Ranking', 'Has IP Address', 'Has At Symbol', 'Short URL', 'Sub Domain Len', 'Hyphen', 'Double Slash', 'URL Length', 'Dir Path Length',
                                        'Digit Letter Ratio', 'Num Digits', 'Num www', 'Query Terms', 'Special Characters']
df_url_data_lexical_features['Label'] = df_url_data['label']

# define the hashing features, later adding back the dependent variable
df_url_data_hashing_features = hashing_feature_pipe.fit_transform(df_url_data.drop(['label'], axis = 1))
df_url_data_hashing_features = pd.DataFrame(df_url_data_hashing_features)
df_url_data_hashing_features.columns = ['Hash' + str(i+1) for i in range(len(df_url_data_hashing_features.columns))]
df_url_data_hashing_features['Label'] = df_url_data['label']

df_url_data_all_features = pd.concat([df_url_data_lexical_features.drop(['Label'], axis = 1), 
                                      df_url_data_hashing_features], axis = 1)

# ----------------------------------------------------------------------------
# STEP 3: EXPLORATORY ANALYTICS ON ENGINEERED FEATURES
# ----------------------------------------------------------------------------

# checking how the ranking distributes
# using a histogram on the ranking variable
sns.histplot(x=np.log(df_url_data_all_features['Alexa Ranking']))

# checking how the length of the url distributes
# using a historgram on the url_length
sns.histplot(x=np.log(df_url_data_all_features['URL Length']))

# scatter plot among all important features in the dataset
sns.pairplot(df_url_data_all_features[['Alexa Ranking', 'URL Length', 'Num www', 'Digit Letter Ratio', 'Dir Path Length', 'Special Characters', 'Label']].sample(frac=0.05), hue="Label")
# scatter plot among some hash features in the dataset
sns.pairplot(df_url_data_hashing_features[['Hash1', 'Hash2', 'Hash3', 'Hash4', 'Hash5', 'Label']].sample(frac=0.05), hue="Label")

# plot some categoricals via a faceted bar plot
cat_data = df_url_data_lexical_features[['Has At Symbol', 'Short URL', 'Hyphen', 'Double Slash', 'Query Terms', 'Label']]
g = sns.catplot(data=cat_data, x='Label', col='Has At Symbol', kind='count')
g.add_legend()

plt.show()

# ----------------------------------------------------------------------------
# STEP 4: EXAMINING MULTI-COLLINEARITY OF FEATURES
# ----------------------------------------------------------------------------

# here we check for multicollinearity in the spearman rank correlation coefficients via hierarchical clustering
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(df_url_data_all_features.drop(['Has IP Address'], axis=1)).correlation
corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(
    corr_linkage, labels=df_url_data_all_features.columns.tolist(), ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro['ivl']))

ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
ax2.set_yticklabels(dendro['ivl'])
fig.tight_layout()
plt.show()

# here we check for associations among categorical fields in the data
# Convert you str columns to Category columns
cat_data = df_url_data_lexical_features[['Has At Symbol', 'Short URL', 'Hyphen', 'Double Slash', 'Query Terms', 'Label']].sample(frac=0.05).apply(
        lambda x: x.astype("category"))

# Initialize a CamresV object using you pandas.DataFrame
cramersv = am.CramersV(cat_data) 
# will return a pairwise matrix filled with Cramer's V, where columns and index are 
# the categorical variables of the passed pandas.DataFrame
cramersvplot = cramersv.fit()

fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
ax1.imshow(cramersvplot, cmap='Greens')
ax1.set_xticks([ i for i in range(len(cat_data.columns))])
ax1.set_yticks([ i for i in range(len(cat_data.columns))])
ax1.set_xticklabels(cat_data.columns, rotation='vertical')
ax1.set_yticklabels(cat_data.columns)
fig.tight_layout()
plt.show()
