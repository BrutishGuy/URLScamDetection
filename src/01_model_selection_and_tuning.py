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


# set pandas options
pd.set_option('display.max_columns', 500)
pd.options.mode.chained_assignment = None

def main():
    # ----------------------------------------------------------------------------
    # STEP 1: ENVIRONMENT CONSTANTS AND DATA LOADING
    # ----------------------------------------------------------------------------
    
    # Model and Program Settings ===
    parser = argparse.ArgumentParser(description='URL Scam Detection Model Selection and Tuning. Use this module to tune and find top performing models on various feature combinations, and to tune the feature hashing trick.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--data_folder', type=str, default='./data/',
                        help='Dataset location folder, detault is data folder in the root directory of repository.')
    parser.add_argument('--output_folder', type=str, default='./output/', help='Model and report output location folder, detault is output folder in the root directory of repository.')
    parser.add_argument('--analysis_folder', type=str, default='./analysis/', help='Analysis location folder, detault is analysis folder in the root directory of repository.')
      
    opt = parser.parse_args()
    DATA_FOLDER = opt.data_folder
    MODEL_OUTPUT_PATH = opt.output_folder
    TRAIN_MODEL_FROM_SCRATCH = True
    MODEL_RUN_DATE = datetime.today() # current date
    ANALYSIS_PATH = opt.analysis_folder
    
    
    # read the data in via gzip and pandas
    url_data = []
    for line in gzip.open(DATA_FOLDER + "urlset.json.gz", 'r'):
        url_data.append(json.loads(line))
        
    df_url_data = pd.DataFrame(url_data)
    
    X = df_url_data[['domain', 'ranking']]
    y = df_url_data.label
    
    # NB the dataset is already very balanced, so we do not take additional measures to balance the data
    
    # ----------------------------------------------------------------------------
    # STEP 2: TRAIN TEST SPLIT, FEATURE ENGINEERING
    # ----------------------------------------------------------------------------
    
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.25, random_state=42)
    
    text_fetcher = FunctionTransformer(get_text_data)
    other_data_fetcher = FunctionTransformer(get_all_other_data)
    log_transformer = FunctionTransformer(log_transform)
    
    # ----------------------------------------------------------------------------
    # STEP 3.1: MODEL SELECTION AND HYPERPARAMETER STEP USING HPSKLEARN
    # ----------------------------------------------------------------------------
    
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
    
    d1 = MODEL_RUN_DATE.strftime("%d_%m_%Y")
    dump(mlpipe, MODEL_OUTPUT_PATH + '_hyperopt_sklearn_url_spam_detection_model_selection_hyperparameter_tuning_' + d1 + '.joblib') 
    
    y_pred = mlpipe.predict(X_test)
    y_pred_proba = mlpipe.predict_proba(X_test)[:, 1]
    
    # ----------------------------------------------------------------------------
    # STEP 3.2: REPORTING
    # ----------------------------------------------------------------------------
    
    df_metrics_temp = pd.DataFrame(columns=['RUN_DATE', 'MODEL', 'ACCURACY', 'F1 SCORE MICRO AVG', 'PRECISION SCORE MICRO AVG', 'RECALL SCORE MICRO AVG', 'ROCAUC MICRO AVG'])
    # print to std out
    # calculate accuracy, f1, precision, recall, ROCAUC.
    df_metrics_temp.loc[0] = [MODEL_RUN_DATE, 
                              model_choice,
                                accuracy_score(y_pred, y_test),
                                f1_score(y_pred, y_test),
                                precision_score(y_pred, y_test),
                                recall_score(y_pred, y_test),
                                roc_auc_score(y_score=y_pred_proba, y_true=y_test)]
    
    
    print(df_metrics_temp)
    
    # ----------------------------------------------------------------------------
    # STEP 4.1: TUNING FEATURE HASHING HYPERPARAMETERS WITH AN XGBOOST MODEL AS THE CLASSIFIER
    # ----------------------------------------------------------------------------
    
    mlpipe.fit(X_train, y_train)
    
    d1 = MODEL_RUN_DATE.strftime("%d_%m_%Y")
    dump(mlpipe, MODEL_OUTPUT_PATH + '_hyperopt_feature_hashing_hyperparameter_tuning_' + d1 + '.joblib') 
    
    y_pred = mlpipe.predict(X_test)
    y_pred_proba = mlpipe.predict_proba(X_test)[:, 1]
    
    # ----------------------------------------------------------------------------
    # STEP 4.2: REPORTING
    # ----------------------------------------------------------------------------
    
    df_metrics_temp = pd.DataFrame(columns=['RUN_DATE', 'MODEL', 'ACCURACY', 'F1 SCORE MICRO AVG', 'PRECISION SCORE MICRO AVG', 'RECALL SCORE MICRO AVG', 'ROCAUC MICRO AVG'])
    # print to std out
    # calculate accuracy, f1, precision, recall, ROCAUC.
    df_metrics_temp.loc[0] = [MODEL_RUN_DATE, 
                              model_choice,
                                accuracy_score(y_pred, y_test),
                                f1_score(y_pred, y_test),
                                precision_score(y_pred, y_test),
                                recall_score(y_pred, y_test),
                                roc_auc_score(y_score=y_pred_proba, y_true=y_test)]
    
    
    print(df_metrics_temp)
    
if __name__ == "__main__":
    main()
