# -*- coding: utf-8 -*-
"""
@author: VictorGueorguiev
"""
from urllib.parse import urlparse
import re
import socket
import whois
import time
import sys
from src.patterns import *
import numpy as np
import pandas as pd

def log_transform(df):
    '''
    Returns log transformed versions of all fields/columns in df. Expects all columns
    to be of int64 of float64 dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe storing the numerical fields to be transformed

    Returns
    -------
    df : pd.DataFrame 
         Dataframe with all fields transformed via a log

    '''
    try:
        result = np.log(df) # scaling numerical fields so that they are more well-distributed for modelling
    except:
        print('Not all columns are numerical!')
    return result


def get_ip_address_checks(df):
    '''
    Returns a dataframe where each domain/URL field is transformed to show whether it contains an IP address in IPv4 or IPv6 format
    
    This function necessarily is designed to be used as part of a FunctionTransformer and Pipeline modelling procedure in Scikit-Learn,
    hence the odd column replacement below, with raw data replaced by feature.
    
    Parameters
    ----------
    df : pd.DataFrame 
        Pandas dataframe storing the text domain/URL column(s) to be featurized

    Returns
    -------
    df : pd.DataFrame
        Pandas dataframe with all domain/URL fields replaced by their feature derivative: 
            In this case a binary 1/0 whether an IP address is present.

    '''
    def is_an_ip_address_check(url):
        '''
        Helper function checks whether a URL string contains any IPv4 or IPv6 addresses
        
        Parameters
        ----------
        url : str
            URL/domain to analyze for presence of IPv4/IPv6 addresses.

        Returns
        -------
        bool
            Binary output of 1 for IP present, 0 for no IP address present in URL string.

        '''
        ip_address_pattern = ipv4_pattern + "|" + ipv6_pattern
        match = re.search(ip_address_pattern, url)
        return 1 if match else 0
    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].astype(str).apply(lambda x: is_an_ip_address_check(x))
    return df


def get_hostname_from_url(url):
    '''
    Returns the hostname of a specified URL string.
    
    Parameters
    ----------
    url : str
        URL to extract the hostname from. e.g. https://www.google.com/search?
        will have hostname www.google.com

    Returns
    -------
    hostname : str
        The hostname of the input URL string.

    '''
    hostname = url
    pattern = "https://|http://|www.|https://www.|http://www."
    pre_pattern_match = re.search(pattern, hostname)

    if pre_pattern_match:
        hostname = hostname[pre_pattern_match.end():]
        post_pattern_match = re.search("/", hostname)
        if post_pattern_match:
            hostname = hostname[:post_pattern_match.start()]

    return hostname

def get_url_lengths(df):
    '''
    Returns a dataframe where each domain/URL field is transformed to extract the length of the URL as no. of characters
    
    This function necessarily is designed to be used as part of a FunctionTransformer and Pipeline modelling procedure in Scikit-Learn,
    hence the odd column replacement below, with raw data replaced by feature.
    
    Parameters
    ----------
    df : pd.DataFrame 
        Pandas dataframe storing the text domain/URL column(s) to be featurized

    Returns
    -------
    df : pd.DataFrame
        Pandas dataframe with all domain/URL fields replaced by their feature derivative: 
            In this case the length of the URL as no. of characters

    '''
    for col in df.columns:
        df.loc[:, col] = np.log(df[col].astype(str).apply(lambda x: len(x)))
    return df

def get_url_length_categories(df):
    def url_length_categorizer(url):
        if len(url) < 54:
            return 1
        if 54 <= len(url) <= 75:
            return 0
        return -1
    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].astype(str).apply(lambda x: url_length_categorizer(x))
    return df


def get_shortening_service_checks(df):
    def shortening_service_check(url):
        match = re.search(shortening_services, url)
        return 1 if match else 0
    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].astype(str).apply(lambda x: shortening_service_check(x))
    return df

def get_at_symbol_checks(df):
    def at_symbol_check(url):
        match = re.search('@', url)
        return 1 if match else 0
    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].astype(str).apply(lambda x: at_symbol_check(x))
    return df


def get_double_slash_positions(df):
    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].astype(str).apply(lambda x: x.count('//'))
    return df


def get_prefix_suffix_positions(df):
    def has_prefix_suffix(domain):
        match = re.search('-', domain)
        return 1 if match else 0
    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].astype(str).apply(lambda x: has_prefix_suffix(x))
    return df


def get_number_of_sub_domain_checks(df):
    def having_ip_address(url):
        ip_address_pattern = ipv4_pattern + "|" + ipv6_pattern
        match = re.search(ip_address_pattern, url)
        return 1 if match else 0

    def has_sub_domain(url):
        # Here, instead of greater than 1 we will take greater than 3 since the greater than 1 condition is when www and
        # country domain dots are skipped
        # Accordingly other dots will increase by 1
        if having_ip_address(url) == 1:
            match = re.search(
                '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
                '([01]?\\d\\d?|2[0-4]\\d|25[0-5]))|(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}',
                url)
            pos = match.end()
            url = url[pos:]
        num_dots = [x.start() for x in re.finditer(r'\.', url)]
        return len(num_dots)
    
    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].astype(str).apply(lambda x: has_sub_domain(x))
    return df

def get_number_of_special_character_checks(df):
    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].astype(str).apply(lambda x: x.count('?')) +\
            df.loc[:, col].astype(str).apply(lambda x: x.count('@')) +\
            df.loc[:, col].astype(str).apply(lambda x: x.count('-')) +\
            df.loc[:, col].astype(str).apply(lambda x: x.count('%')) +\
            df.loc[:, col].astype(str).apply(lambda x: x.count('.')) +\
            df.loc[:, col].astype(str).apply(lambda x: x.count('='))
    return df

def get_number_of_query_term_checks(df):
    for col in df.columns:
        df.loc[:, col] =  df.loc[:, col].astype(str).apply(lambda x: x.count('%20')) 
    return df

def get_number_of_www_checks(df):
    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].astype(str).apply(lambda x: x.count('www'))
    return df

def get_number_of_digits(df):
    def digit_count(url):
        digits = 0
        for i in url:
            if i.isnumeric():
                digits = digits + 1
        return digits
    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].astype(str).apply(lambda x: digit_count(x))
    return df


def get_digit_to_letter_ratio(df):
    def letter_count(url):
        letters = 0
        for i in url:
            if i.isalpha():
                letters = letters + 1
        return letters
    def digit_count(url):
        digits = 0
        for i in url:
            if i.isnumeric():
                digits = digits + 1
        return digits
    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].astype(str).apply(lambda x: digit_count(x))/df.loc[:, col].astype(str).apply(lambda x: letter_count(x))
    return df


def get_number_of_dir_paths(df):
    def no_of_dir(url):
        urldir = urlparse(url).path
        return urldir.count('/')
    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].astype(str).apply(lambda x: no_of_dir(x))
    return df

