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
import jax

def setup(cfg):
    cfg.EXP = cfg.NAME
    cfg.INPUT = cfg.DATA_PATH # path of source files
    cfg.OUTPUT = os.path.join(cfg.MAIN_PATH, "output") # path of output files

    cfg.OUTPUT_EXP = os.path.join(cfg.OUTPUT, cfg.EXP)
    cfg.EXP_MODEL = os.path.join(cfg.OUTPUT_EXP, "model")
    cfg.EXP_FIG = os.path.join(cfg.OUTPUT_EXP, "fig")
    cfg.EXP_LOG = os.path.join(cfg.OUTPUT_EXP, "log")
    cfg.EXP_PREDS = os.path.join(cfg.OUTPUT_EXP, "preds")

    # create directories for output files
    for d in [cfg.EXP_MODEL, cfg.EXP_FIG, cfg.EXP_LOG,cfg.EXP_PREDS]:
        os.makedirs(d, exist_ok=True)
    return cfg

# 乱数固定用の関数
def set_seed(seed=42):
    random.seed(seed) # seed for python
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # seed for numpy


def abbrev_alg(alg_name):
    
    abbrev_dict = {
        'logistic':'LR',
        'random_forest':'RF',
        'adaboost':'AdaBoost',
        'mlp1221':'MLP',
    }

    return abbrev_dict[alg_name]

def show_score_name(score_name):
    show_score_dict = {
        'accuracy_score': 'Accuracy',
        'f1_score': 'F1',
        'f1_score_macro': 'F1 (macro)',
        'recall_score': 'Recall',
        'recall_score_macro': 'Recall (macro)',
        'precision_score': 'Precision',
        'precision_score_macro': 'Precision (macro)',
        'cross_entropy': 'Cross Entropy',
        'entropy': 'Entropy'
    }

def show_score_abbrev(score_name):
    show_score_dict = {
        'accuracy_score': 'Acc',
        'f1_score': 'F1',
        'f1_score_macro': 'F1',
        'recall_score': 'Rec',
        'recall_score_macro': 'Rec',
        'precision_score': 'Prec',
        'precision_score_macro': 'Prec',
        'cross_entropy': 'CE',
        'entropy': 'SE'
    }

    return show_score_dict[score_name]

def show_dataset_name(dataset_name):
    show_dataset_dict = {
        'bank':'Bank',
        'adult': 'Adult',
    }

    return show_dataset_dict[dataset_name]

def show_param_name(param_name):
    show_param_dict ={
        'round_alpha': r'$\gamma$',
        'k': 'k',
    }

    return show_param_dict[param_name]

def show_method_name(method_name):
    show_method_dict={
        'ord': 'ord',
        'comp': 'comp',
        'propose': 'proposed',
        'IPAL': 'IPAL',
    }
    return show_method_dict[method_name]


def return_task(dataset_name):

    if dataset_name in ['bank', 'adult']: 
        return 'classification'
    else:
        return 'regression'