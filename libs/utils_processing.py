import os
import gc
import re
import sys
import json
import time
import shutil
import joblib
import random
import requests
import pickle
import arff
import warnings
warnings.filterwarnings('ignore')
from ast import literal_eval
import argparse

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from utils import *
from evaluation import *

# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
def exec_ohe(df, ohe_cols, is_comp=False):
    '''
    apply onehot encoding to pd.DataFrame df
    Args:
        df: pd.DataFrame
        ohe_cols: list
            list of names of columns to be one hot encoded
        is_comp: bool
            When it is True, ohe_cols are treated as CFs
    Returns: 
        output_df: pd.DataFrame
    '''
    ohe = OneHotEncoder(sparse=False, categories='auto')
    ohe.fit(df[ohe_cols])

    tmp_columns = []
    for i, col in enumerate(ohe_cols):
        tmp_columns += [f'{col}_{v}' for v in ohe.categories_[i]]
    
    df_tmp = pd.DataFrame(ohe.transform(df[ohe_cols]), columns=tmp_columns)
    
    # if the features are represented as complementary label, the value of the index assigned to the complementary label should be 0
    if is_comp: 
        df_tmp = df_tmp * (-1) + 1
    output_df = pd.concat([df.drop(ohe_cols, axis=1), df_tmp], axis=1)

    return output_df


def privacy_transform(df, mask_feature_list, mode="comp", seed=42):
    '''
    Transforming OF to CF due to privacy concerns
    Args:
        df: pd.DataFrame
            original data (not one-hot encoded)
        mask_feature_list: list
            list of feature names to be complementary transformed
        mode: str
            comp:complementary, partial:partial
    Return
        transformed_df: pd.DataFrame 
    '''


    set_seed(seed)

    output_df = df.copy()

    if mode == "comp":
        for comp_feature_name in mask_feature_list:
            val_list = output_df[comp_feature_name].unique().tolist() # list of unique values of the column
        
            # select complementary value uniformly
            #output_df[comp_feature_name] = output_df[comp_feature_name].map(lambda x: random.choice(sorted(list(set(val_list) - set([x])))) )
            for i in range(output_df.shape[0]):
                output_df[comp_feature_name].iloc[i] = random.choice(sorted(list(set(val_list) - set([output_df[comp_feature_name].iloc[i]]))))
    elif mode == "partial":
        raise NotImplementedError
    

    return output_df

def test_random_choice(seed):
    random.seed(seed)
    values = [1, 2, 3, 4, 5]
    result = [random.choice(values) for _ in range(10)]
    return result

def hard_labeling(df, comp_cols):
    '''
    Hard labeling OneHot encoded categorical features
    Args:
        df: pd.DataFrame
            This df inludes OneHot encoded categorical features
        comp_cols: list
            list of original names of OneHot encoded categorical features
    '''
    output_df = df.copy()
    N = df.shape[0]

    for col in comp_cols:
        col_onehot_list = [c for c in df.columns.tolist() if col in c]
        soft_labels = df.loc[:, col_onehot_list].values
        hard_labels = np.zeros(soft_labels.shape, dtype=np.float32)

        argmax_list = np.argmax(soft_labels, axis=1).tolist()

        for i in range(N):
            hard_labels[i, argmax_list[i]] = 1
    
        output_df.loc[:, col_onehot_list] = hard_labels
    
    return output_df

