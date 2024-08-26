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
import yaml
import argparse
import warnings
warnings.filterwarnings('ignore')
from ast import literal_eval
import argparse
import logging
from logging import getLogger, Logger
from pytz import timezone
from datetime import datetime
import math
import copy
import gc

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

import jax

from sklearn.model_selection import train_test_split


sys.path.append("./libs")
from utils import * 
from evaluation import *
from load_data import *
from learning import *
from methods import *
from utils_processing import *
from helpers import *

def run(args, logger):

    # set seed
    set_seed(args.seed)

    # load data
    logger.info("data loading...")
    data_df, cat_cols = load_data(data_name=args.dataset_name, data_path=args.data_dir,
                                  sample_size=args.sample_size, seed=args.seed)
    
    if args.task == 'classification':
        n_classes = data_df['target'].nunique()
    else:
        n_classes = None

    # convert some ordinary features to complementary features
    if 'all' in args.comp_cols:
        comp_feature_list = cat_cols
    else:
        comp_feature_list = args.comp_cols

    comp_data_df = privacy_transform(df=data_df, mask_feature_list=comp_feature_list, mode="comp", seed=args.seed)
    
    # one hot encoding
    data_df = exec_ohe(data_df, cat_cols, is_comp=False)

    cat_cols_ord = [c for c in cat_cols if c not in comp_feature_list]
    comp_data_df = exec_ohe(comp_data_df, cat_cols_ord, is_comp=False) 
    comp_data_df = exec_ohe(comp_data_df, comp_feature_list, is_comp=True)

    comp_onehot_names_list = []
    # comp OneHot column names
    for cat_col in comp_feature_list:
        comp_onehot_names_list += [col for col in comp_data_df.columns.tolist() if cat_col in col]

    # normalization of comp cols
    col_names = comp_data_df.columns.tolist()
    for col in comp_feature_list:
        col_ohe_idx = [i for i, c in enumerate(col_names) if col in c]
        comp_data_df.iloc[:, col_ohe_idx] *= 1.0/(len(col_ohe_idx) -1)

    avoid_estimate_cols_ohe = [] # List of column names in onehot vector of CF not estimated
    # Limit comp_feature_list here to limit CFs to be estiamted
    if len(args.avoid_estimate_cols) >= 1:
        comp_feature_list = list(set(comp_feature_list) - set(args.avoid_estimate_cols))
        for avoid_col in args.avoid_estimate_cols:
            avoid_estimate_cols_ohe.extend([col for col in comp_data_df.columns.tolist() if avoid_col in col])

    # train test split
    test_index = sorted(random.sample(data_df.index.tolist(), int(data_df.shape[0] * args.test_rate)))
    train_index = sorted(list(set(data_df.index.tolist()) - set(test_index)))

    ord_train_df = data_df.iloc[train_index].reset_index(drop=True)
    ord_test_df = data_df.iloc[test_index].reset_index(drop=True)
    comp_train_df = comp_data_df.iloc[train_index].reset_index(drop=True)
    comp_test_df = comp_data_df.iloc[test_index].reset_index(drop=True)


    # training and evaluation
    ## for log
    train_model_score_dict = {} # Save the model's prediction score when using data where the CFs' exact values is estimated only from the training data.
    all_model_score_dict = {} # Save the model's prediction score when using data where the CFs' exact values is estimated from the training data + test_data.
    train_disamb_score_dict = {} # Save the score of estimation for CF's exact values when using data where the CFs' exact values is estimated only from the training data.
    all_disamb_score_dict = {} # Save the score of estimation for CF's exact values when using data where the CFs' exact values is estimated from the training data + test data.

    if args.method in ["propose"]:
        train_model_score_dict= {'soft': {}, 'hard':{}}
        all_model_score_dict= {'soft': {}, 'hard':{}}
        train_model_score_only_comp_dict= {'soft': {}, 'hard':{}}
        all_model_score_only_comp_dict= {'soft': {}, 'hard':{}}
    elif args.method == "IPAL":
        train_model_score_dict= {'soft': {}, 'CMN_hard':{}, 'origin': {}}
        all_model_score_dict= {'soft': {},  'CMN_hard':{}}
        train_model_score_only_comp_dict= {'soft': {}, 'CMN_hard':{}, 'origin':{}}
        all_model_score_only_comp_dict= {'soft': {}, 'CMN_hard':{}}

    # disambiguation (estimating CFs' exact values)=========================================================================================================================================
    logger.info("disambiguation...")
    if args.method == "propose":
        # disambiguation (train only)
        with jax.default_device(jax.devices('cpu')[0]):
            train_output_df = propose_method(df=comp_train_df.drop(['target']+avoid_estimate_cols_ohe, axis=1), 
                                                                        comp_cols=comp_feature_list, 
                                                                        cat_cols=cat_cols, 
                                                                        knn_metric=args.knn_metric, 
                                                                        correct_own_comp=args.correct_own_comp,
                                                                        k=args.k, 
                                                                        max_prop_iter=args.max_prop_iter, 
                                                                        use_CMN=False,
                                                                        round_max=args.round_max, 
                                                                        round_alpha=args.round_alpha, 
                                                                        n_parallel=args.n_parallel,
                                                                        measure_time=args.measure_time, 
                                                                        use_jax=args.use_jax,
                                                                        logger=logger)
        
        train_output_df = pd.concat([train_output_df, comp_train_df[avoid_estimate_cols_ohe+['target']] ], axis=1)
        
        train_output_hard_df = train_output_df.copy()
        for comp_col in comp_feature_list:
            confs = train_output_hard_df[[col for col in train_output_hard_df.columns.tolist() if comp_col in col]].values
            train_output_hard_df[[col for col in train_output_hard_df.columns.tolist() if comp_col in col]] = np.eye(confs.shape[1])[np.argmax(confs, axis=1)]    
        
        # disambiguation (all data)
        with jax.default_device(jax.devices('cpu')[0]):
            all_output_df = propose_method(df=comp_data_df.drop(['target']+avoid_estimate_cols_ohe, axis=1), 
                                                                    comp_cols=comp_feature_list, 
                                                                    cat_cols=cat_cols, 
                                                                    knn_metric=args.knn_metric, 
                                                                    correct_own_comp=args.correct_own_comp,
                                                                    k=args.k, 
                                                                    max_prop_iter=args.max_prop_iter, 
                                                                    use_CMN=False,
                                                                    round_max=args.round_max, 
                                                                    round_alpha=args.round_alpha, 
                                                                    n_parallel=args.n_parallel,
                                                                    measure_time=args.measure_time, 
                                                                    use_jax=args.use_jax,
                                                                    logger=logger,)
        
        all_output_df = pd.concat([all_output_df, comp_data_df[avoid_estimate_cols_ohe+['target']] ], axis=1)
        
        all_output_hard_df = all_output_df.copy()
        for comp_col in comp_feature_list:
            confs = all_output_hard_df[[col for col in all_output_hard_df.columns.tolist() if comp_col in col]].values
            all_output_hard_df[[col for col in all_output_hard_df.columns.tolist() if comp_col in col]] = np.eye(confs.shape[1])[np.argmax(confs, axis=1)]


        # evaluation of disambiguation
        train_disamb_scores_per_col, train_disamb_scores_average = evaluation_disamb_cls(df_true=data_df.iloc[train_index],
                                                                                        df_pred_prob=train_output_df,
                                                                                        df_pred_label=train_output_df,
                                                                                        comp_cols=comp_feature_list,)
        
        all_disamb_scores_per_col_train, all_disamb_scores_average_train = evaluation_disamb_cls(df_true=data_df.iloc[train_index],
                                                                                                df_pred_prob=all_output_df.iloc[train_index],
                                                                                                df_pred_label=all_output_df.iloc[train_index],
                                                                                                comp_cols=comp_feature_list,)
        
        all_disamb_scores_per_col_all, all_disamb_scores_average_all = evaluation_disamb_cls(df_true=data_df,
                                                                                            df_pred_prob=all_output_df,
                                                                                            df_pred_label=all_output_df,
                                                                                            comp_cols=comp_feature_list,)
        
        all_disamb_scores_per_col_test, all_disamb_scores_average_test = evaluation_disamb_cls(df_true=data_df.iloc[test_index],
                                                                                            df_pred_prob=all_output_df.iloc[test_index],
                                                                                            df_pred_label=all_output_df.iloc[test_index],
                                                                                            comp_cols=comp_feature_list,)
        
        train_disamb_score_dict['scores_per_col'] = train_disamb_scores_per_col.copy()
        train_disamb_score_dict['scores_average'] = train_disamb_scores_average.copy()
        all_disamb_score_dict['scores_per_col (train)'] = all_disamb_scores_per_col_train.copy()
        all_disamb_score_dict['scores_average (train)'] = all_disamb_scores_average_train.copy()
        all_disamb_score_dict['scores_per_col (test)'] = all_disamb_scores_per_col_test.copy()
        all_disamb_score_dict['scores_average (test)'] = all_disamb_scores_average_test.copy()
        all_disamb_score_dict['scores_per_col (all)'] = all_disamb_scores_per_col_all.copy()
        all_disamb_score_dict['scores_average (all)'] = all_disamb_scores_average_all.copy()

        del train_disamb_scores_average['confusion_matrix']
        del all_disamb_scores_average_train['confusion_matrix']
        del all_disamb_scores_average_test['confusion_matrix']
        del all_disamb_scores_average_all['confusion_matrix']
        logger.info(f"[train] scores_average: {train_disamb_scores_average}")
        logger.info(f"[all] scores_average (train): {all_disamb_scores_average_train}")
        logger.info(f"[all] scores_average (test): {all_disamb_scores_average_test}")
        logger.info(f"[all] scores_average (all): {all_disamb_scores_average_all}")

        
    elif args.method == 'IPAL':
        train_output_df = comp_data_df.copy() # for estimated confidences
        train_output_CMN_hard_df = comp_data_df.copy()
        all_output_df = comp_data_df.copy() # for estimated confidences
        all_output_CMN_hard_df = comp_data_df.copy()

        # disambiguation (train only)
        with jax.default_device(jax.devices('cpu')[0]):
            IPAL_base = IPAL(k=args.k, alpha=args.ipal_alpha, T=args.max_prop_iter, use_jax=args.use_jax, n_parallel=args.n_parallel)
            logger.info("creating graph...")
            IPAL_base.create_graph(instances=comp_train_df.drop(['target'] + comp_onehot_names_list, axis=1).values.copy())
            for comp_col in comp_feature_list:
                if comp_col in args.avoid_estimate_cols:
                    continue
                logger.info(f"disambiguating '{comp_col}' ...")
                IPAL_tmp = copy.copy(IPAL_base)
                train_pred_ohe_labels, train_pred_confs = IPAL_tmp.predict_transductive(target_confidences = comp_train_df[[col for col in comp_train_df.columns.tolist() if comp_col in col]].values.copy(), 
                                                                                        return_conf=True)
                test_pred_ohe_labels = IPAL_tmp.predict_inductive(new_instances = comp_test_df.drop(['target'] + comp_onehot_names_list, axis=1).values.copy(),
                                                                )
                
                train_output_df.loc[train_index, [col for col in comp_train_df.columns.tolist() if comp_col in col]] = train_pred_confs # only instances with train_index have estimted confidences, whereas test isntances have initial confidences
                # In IPAL, train_output_CMN_hard_df includes the estimated values for test data.
                train_output_CMN_hard_df.loc[train_index, [col for col in comp_train_df.columns.tolist() if comp_col in col]] = train_pred_ohe_labels
                train_output_CMN_hard_df.loc[test_index, [col for col in comp_train_df.columns.tolist() if comp_col in col]] = test_pred_ohe_labels

                del IPAL_tmp
                gc.collect()

            # disambiguation (all data)
            IPAL_base = IPAL(k=args.k, alpha=args.ipal_alpha, T=args.max_prop_iter, use_jax=args.use_jax, n_parallel=args.n_parallel)
            IPAL_base.create_graph(instances=comp_data_df.drop(['target'] + comp_onehot_names_list, axis=1).values.copy())
            for comp_col in comp_feature_list:
                if comp_col in args.avoid_estimate_cols:
                    continue
                IPAL_tmp = copy.copy(IPAL_base)
                all_pred_ohe_labels, all_pred_confs = IPAL_tmp.predict_transductive(target_confidences=comp_data_df[[col for col in comp_data_df.columns.tolist() if comp_col in col]].values.copy(),
                                                                                    return_conf=True)

                all_output_df[[col for col in comp_train_df.columns.tolist() if comp_col in col]] = all_pred_confs
                all_output_CMN_hard_df[[col for col in comp_train_df.columns.tolist() if comp_col in col]] = all_pred_ohe_labels

                del IPAL_tmp
                gc.collect()

        # evaluation of disambiguation
        ## inductive (create graph using only train data)
        train_disamb_scores_per_col_train, train_disamb_scores_average_train = evaluation_disamb_cls(df_true=data_df.iloc[train_index],
                                                                                                    df_pred_prob=train_output_df.iloc[train_index],
                                                                                                    df_pred_label=train_output_CMN_hard_df.iloc[train_index],
                                                                                                    comp_cols=comp_feature_list)
        train_disamb_scores_per_col_test, train_disamb_scores_average_test = evaluation_disamb_cls(df_true=data_df.iloc[test_index],
                                                                                                    df_pred_prob=train_output_CMN_hard_df.iloc[test_index], # for calculating Cross Entropy
                                                                                                    df_pred_label=train_output_CMN_hard_df.iloc[test_index],
                                                                                                    comp_cols=comp_feature_list)
        
        ## transductive (create graph using all data)
        all_disamb_scores_per_col_train, all_disamb_scores_average_train = evaluation_disamb_cls(df_true=data_df.iloc[train_index],
                                                                                                df_pred_prob=all_output_df.iloc[train_index],
                                                                                                df_pred_label=all_output_CMN_hard_df.iloc[train_index],
                                                                                                comp_cols=comp_feature_list)
        all_disamb_scores_per_col_test, all_disamb_scores_average_test = evaluation_disamb_cls(df_true=data_df.iloc[test_index],
                                                                                                df_pred_prob=all_output_df.iloc[test_index],
                                                                                                df_pred_label=all_output_CMN_hard_df.iloc[test_index],
                                                                                                comp_cols=comp_feature_list)
        all_disamb_scores_per_col_all, all_disamb_scores_average_all = evaluation_disamb_cls(df_true=data_df,
                                                                                            df_pred_prob=all_output_df,
                                                                                            df_pred_label=all_output_CMN_hard_df,
                                                                                            comp_cols=comp_feature_list)
        

        train_disamb_score_dict['scores_per_col (train)'] = train_disamb_scores_per_col_train.copy()
        train_disamb_score_dict['scores_average (train)'] = train_disamb_scores_average_train.copy()
        train_disamb_score_dict['scores_per_col (test)'] = train_disamb_scores_per_col_test.copy()
        train_disamb_score_dict['scores_average (test)'] = train_disamb_scores_average_test.copy()

        all_disamb_score_dict['scores_per_col (train)'] = all_disamb_scores_per_col_train.copy()
        all_disamb_score_dict['scores_average (train)'] = all_disamb_scores_average_train.copy()
        all_disamb_score_dict['scores_per_col (test)'] = all_disamb_scores_per_col_test.copy()
        all_disamb_score_dict['scores_average (test)'] = all_disamb_scores_average_test.copy()
        all_disamb_score_dict['scores_per_col (all)'] = all_disamb_scores_per_col_all.copy()
        all_disamb_score_dict['scores_average (all)'] = all_disamb_scores_average_all.copy()

        del train_disamb_scores_average_train['confusion_matrix']
        del train_disamb_scores_average_test['confusion_matrix']
        del all_disamb_scores_average_train['confusion_matrix']
        del all_disamb_scores_average_test['confusion_matrix']
        del all_disamb_scores_average_all['confusion_matrix']

        logger.info(f"[inductive-train] scores_average: {train_disamb_scores_average_train}")
        logger.info(f"[inductive-test] scores_average: {train_disamb_scores_average_test}")
        logger.info(f"[transductive-train] scores_average: {all_disamb_scores_average_train}")
        logger.info(f"[transductive-test] scores_average: {all_disamb_scores_average_test}")
        logger.info(f"[transductive-all] scores_average: {all_disamb_scores_average_all}")

        
    elif args.method == 'comp': # evaluate disambiguation scores for comparison with proposed method
        # evaluation using only training data
        train_disamb_scores_per_col, train_disamb_scores_average = evaluation_disamb_cls(df_true=data_df.iloc[train_index],
                                                                                        df_pred_prob=comp_data_df.iloc[train_index],
                                                                                        comp_cols=comp_feature_list,
                                                                                        labeling_strategy='random')    
        
        # evaluation using only test data
        test_disamb_scores_per_col, test_disamb_scores_average = evaluation_disamb_cls(df_true=data_df.iloc[test_index],
                                                                                        df_pred_prob=comp_data_df.iloc[test_index],
                                                                                        comp_cols=comp_feature_list,
                                                                                        labeling_strategy='random')    

        # evaluation using all data       
        all_disamb_scores_per_col, all_disamb_scores_average = evaluation_disamb_cls(df_true=data_df,
                                                                                    df_pred_prob=comp_data_df,
                                                                                    comp_cols=comp_feature_list,
                                                                                    labeling_strategy='random')
        
        train_disamb_score_dict['scores_per_col'] = train_disamb_scores_per_col.copy()
        train_disamb_score_dict['scores_average'] = train_disamb_scores_average.copy()
        all_disamb_score_dict['scores_per_col'] = all_disamb_scores_per_col.copy()
        all_disamb_score_dict['scores_average'] = all_disamb_scores_average.copy()

        all_disamb_score_dict['scores_per_col (train)'] = train_disamb_scores_per_col.copy()
        all_disamb_score_dict['scores_average (train)'] = train_disamb_scores_average.copy()
        all_disamb_score_dict['scores_per_col (test)'] = test_disamb_scores_per_col.copy()
        all_disamb_score_dict['scores_average (test)'] = test_disamb_scores_average.copy()
        all_disamb_score_dict['scores_per_col (all)'] = all_disamb_scores_per_col.copy()
        all_disamb_score_dict['scores_average (all)'] = all_disamb_scores_average.copy()

        del train_disamb_scores_average['confusion_matrix']
        del all_disamb_scores_average['confusion_matrix']
        del test_disamb_scores_average['confusion_matrix']
        logger.info(f"[train] scores_average: {train_disamb_scores_average}")
        logger.info(f"[test] scores_average: {test_disamb_scores_average}")
        logger.info(f"[all] scores_average (train): {all_disamb_scores_average}")
           

    if args.task == "classification":
        algorithm_list = ['logistic', 'random_forest', 'adaboost', 'mlp1221']
    elif args.task == "regression":
        raise NotImplementedError
    else:
        raise NotImplementedError

    # prediction of output labels ================================================================================================================================
    logger.info("training and evaluation with all columns")

    for algorithm in algorithm_list:
        logger.info(f"algorithm = {algorithm}")

        if args.method == "ord":
            train_model = training(train_df=ord_train_df, target_col='target', algorithm=algorithm, task=args.task, seed=args.seed)
            scores = evaluating(test_df=ord_test_df, model=train_model, target_col='target', task=args.task, n_classes=n_classes)
            train_model_score_dict[algorithm] = scores.copy()

            del scores['confusion_matrix']
            logger.info(scores)

        elif args.method == "comp":
            train_model = training(train_df=comp_train_df, target_col='target', algorithm=algorithm, task=args.task, seed=args.seed)
            scores = evaluating(test_df=comp_test_df, model=train_model, target_col='target', task=args.task, n_classes=n_classes)
            train_model_score_dict[algorithm] = scores.copy()

            del scores['confusion_matrix']
            logger.info(scores)
        
        elif args.method == "IPAL":
            train_model = training(train_df=train_output_df.loc[train_index, :], target_col='target', algorithm=algorithm, task=args.task, seed=args.seed)
            train_CMN_hard_model = training(train_df=train_output_CMN_hard_df.loc[train_index, :], target_col='target', algorithm=algorithm, task=args.task, seed=args.seed)
            all_model = training(train_df=train_output_df.loc[train_index, :], target_col='target', algorithm=algorithm, task=args.task, seed=args.seed)
            all_CMN_hard_model = training(train_df=all_output_CMN_hard_df.loc[train_index, :], target_col='target', algorithm=algorithm, task=args.task, seed=args.seed)

            train_scores = evaluating(test_df=all_output_df.loc[test_index, :], model=train_model, target_col='target', task=args.task, n_classes=n_classes)
            train_CMN_hard_scores = evaluating(test_df=all_output_CMN_hard_df.loc[test_index, :], model=train_CMN_hard_model, target_col='target', task=args.task, n_classes=n_classes)
            train_origin_scores = evaluating(test_df=train_output_CMN_hard_df.loc[test_index, :], model=train_CMN_hard_model, target_col='target', task=args.task, n_classes=n_classes)
            all_scores = evaluating(test_df=all_output_df.loc[test_index, :], model=all_model, target_col='target', task=args.task, n_classes=n_classes)
            all_CMN_hard_scores = evaluating(test_df=all_output_CMN_hard_df.loc[test_index, :], model=all_CMN_hard_model, target_col='target', task=args.task, n_classes=n_classes)

            train_model_score_dict['soft'][algorithm] = train_scores.copy()
            train_model_score_dict['CMN_hard'][algorithm] = train_CMN_hard_scores.copy()
            train_model_score_dict['origin'][algorithm] = train_origin_scores.copy()
            all_model_score_dict['soft'][algorithm] = all_scores.copy()
            all_model_score_dict['CMN_hard'][algorithm] = all_CMN_hard_scores.copy()

            del train_scores['confusion_matrix']
            del train_CMN_hard_scores['confusion_matrix']
            del train_origin_scores['confusion_matrix']
            del all_scores['confusion_matrix']
            del all_CMN_hard_scores['confusion_matrix']
            logger.info(f"train_cores: {train_scores}")
            logger.info(f"train_CMN_hard_scores: {train_CMN_hard_scores}")
            logger.info(f"train_origin_scores: {train_origin_scores}")
            logger.info(f"all_scores: {all_scores}")
            logger.info(f"all_CMN_hard_scores: {all_CMN_hard_scores}")

        elif args.method in ["propose"]:
            train_model = training(train_df=train_output_df, target_col='target', algorithm=algorithm, task=args.task, seed=args.seed)
            train_hard_model = training(train_df=train_output_hard_df, target_col='target', algorithm=algorithm, task=args.task, seed=args.seed)
            all_model = training(train_df=all_output_df.loc[train_index, :], target_col='target', algorithm=algorithm, task=args.task, seed=args.seed)
            all_hard_model = training(train_df=all_output_hard_df.loc[train_index, :], target_col='target', algorithm=algorithm, task=args.task, seed=args.seed)

            train_scores = evaluating(test_df=all_output_df.loc[test_index, :].reset_index(drop=True), model=train_model, target_col='target', task=args.task, n_classes=n_classes)
            train_hard_scores = evaluating(test_df=all_output_hard_df.loc[test_index, :].reset_index(drop=True), model=train_hard_model, target_col='target', task=args.task, n_classes=n_classes)
            all_scores = evaluating(test_df=all_output_df.loc[test_index, :].reset_index(drop=True), model=all_model, target_col='target', task=args.task, n_classes=n_classes)
            all_hard_scores = evaluating(test_df=all_output_hard_df.loc[test_index, :].reset_index(drop=True), model=all_hard_model, target_col='target', task=args.task, n_classes=n_classes)

            train_model_score_dict['soft'][algorithm] = train_scores.copy()
            train_model_score_dict['hard'][algorithm] = train_hard_scores.copy()
            all_model_score_dict['soft'][algorithm] = all_scores.copy()
            all_model_score_dict['hard'][algorithm] = all_hard_scores.copy()

            del train_scores['confusion_matrix']
            del train_hard_scores['confusion_matrix']
            del all_scores['confusion_matrix']
            del all_hard_scores['confusion_matrix']
            logger.info(f"train_scores: {train_scores}")
            logger.info(f"train_hard_scores: {train_hard_scores}")
            logger.info(f"all_scores: {all_scores}")
            logger.info(f"all_hard_scores: {all_hard_scores}")
            
    
    # save logs
    log_df_dict = {}
    if args.method == 'propose':
        log_df_dict['train_output_df'] = train_output_df
        log_df_dict['train_output_hard_df'] = train_output_hard_df
        log_df_dict['all_output_df'] = all_output_df
        log_df_dict['all_output_hard_df'] = all_output_hard_df
    elif args.method == 'IPAL':
        log_df_dict['train_output_df'] = train_output_df
        log_df_dict['train_output_CMN_hard_df'] = train_output_CMN_hard_df
        log_df_dict['all_output_df'] = all_output_df
        log_df_dict['all_output_CMN_hard_df'] = all_output_CMN_hard_df  

    log_dict = {}
    log_dict['args'] = args
    log_dict['train_model_score'] = train_model_score_dict
    log_dict['all_model_score'] = all_model_score_dict

    log_dict['train_disamb_score'] = train_disamb_score_dict
    log_dict['all_disamb_score'] = all_disamb_score_dict

    log_name = get_log_filename(args)
    with open(os.path.join(args.output_dir, args.exp_name, log_name, log_name +"_df.pkl"), "wb") as f:
        pickle.dump(log_df_dict, f)
    with open(os.path.join(args.output_dir, args.exp_name, log_name, log_name +"_log.pkl"), "wb") as f:
        pickle.dump(log_dict, f)
    


if __name__ == "__main__":

    parser = arg_parser()
    args = parser.parse_args()
    print(args)
    if args.config_file is not None and os.path.exists(args.config_file):
        config_file = args.config_file
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config_args = argparse.Namespace(**config)
            for k, v in config_args.__dict__.items():
                if v is not None:
                    setattr(args, k, v)
        print(f"Loaded config from {config_file}!")

    # load default config
    default_config_dict = get_args_default()
    for key, val in config_args.__dict__.items():
        if key not in default_config_dict.keys():
            setattr(args, key, val)
    

    if args.dataset_name in ['bank', 'adult']:
        args.task = 'classification'
    else:
        args.task = 'regression'

    args.exp_name += f"_{args.dataset_name}"

    log_name = get_log_filename(args)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.exp_name), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.exp_name, log_name), exist_ok=True)
    

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        #filename=os.path.join(args.output_dir, args.exp_name, log_filename),
    )

    logger=getLogger(args.dataset_name)

    # https://qiita.com/r1wtn/items/d615f19e338cbfbfd4d6
    # Set handler to output to files
    fh = logging.FileHandler(os.path.join(args.output_dir, args.exp_name, log_name, log_name + ".log"))
    fh.setLevel(logging.DEBUG)
    def customTime(*args):
        return datetime.now(timezone('Asia/Tokyo')).timetuple()
    formatter = logging.Formatter(
        fmt='%(levelname)s : %(asctime)s : %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S %z"
    )
    formatter.converter = customTime
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
   
    # logging args
    for k, v in config_args.__dict__.items():
        logger.info(f"args[{k}] = {v}")

    run(args, logger)




