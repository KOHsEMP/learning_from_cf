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

import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import jax
import jaxopt
from jax import jit
import jax.numpy as jnp
import jax.scipy as jsp


from utils import * 
from evaluation import *


# learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss)

def training(train_df, target_col, algorithm, task, seed):
    '''
    Training label prediction model
    Args:
        train_df: training data (pd.DataFrame) includes input features (instances) and output feature (output label)
        target_col: output feature name
        algorithm: training algorithm for creating label prediction model (choices: "logistic", "random_forest", "adaboost", "mlp1221")
        task: "classification" ("regression" is not implemented)
        seed: random seed for training algorithms
    Returns:
        model: trained label prediction model
    '''

    # training
    if algorithm == "logistic":
        model = LogisticRegression(random_state=seed, solver="lbfgs", multi_class='auto', n_jobs=1)    
    elif algorithm == "random_forest":
        if task == "classification":
            model = RandomForestClassifier(random_state=seed, n_jobs=1)
        else: # regression
            raise NotImplementedError
    elif algorithm == "adaboost":
        if task == "classification":
            model = AdaBoostClassifier(random_state=seed,
                                    n_estimators=10,
                                    base_estimator= DecisionTreeClassifier(max_depth=10))
        else: # regression
            raise NotImplementedError
    
    elif algorithm == "mlp1221":
        if task == "classification":
            model = MLPClassifier(hidden_layer_sizes=(100, 200, 200, 100), activation='relu', solver='adam',
                                shuffle=True, random_state=seed, early_stopping=True)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    model.fit(train_df.drop(target_col,axis=1), train_df[target_col])  
    
    return model
    

def evaluating(test_df, model, target_col, task, n_classes=None):
    '''
    Evaluating the prediction results of the givin label prediction model.
    Args:
        test_df: test data (pd.DataFrame) includes input features (instances) and output feature (output label)
        model: label prediction model (by created "training" function)
        target_col: output feature name
        n_classes: the number of classes when the problem is classification problem
    Returns:
        scores: dict (key: score name, val: score value)
    '''

    if task == "classification":
        test_pred_label = model.predict(test_df.drop(target_col,axis=1))
        test_pred_prob  = model.predict_proba(test_df.drop(target_col,axis=1))
    elif task == "regression":
        raise NotImplementedError

    else:
        raise NotImplementedError
    
    # evaluate
    if task == "classification":
        scores = evaluation_classification_detail(true_value=test_df[target_col].values, 
                                                    pred_label=test_pred_label, 
                                                    pred_prob=test_pred_prob,
                                                    n_classes=n_classes)
    elif task == "regression":
        raise NotImplementedError
    else:
        raise NotImplementedError
    

    return scores
