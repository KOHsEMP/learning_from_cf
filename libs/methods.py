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


from scipy.spatial import distance
from scipy.optimize import minimize
from jax.scipy.optimize import minimize as jax_minimize
from joblib import Parallel, delayed
from pathos.multiprocessing import ProcessingPool
from pathos.pools import ProcessPool

@jit
def jax_euclidean_dist_matrix(X, Y):
    X_sum = jnp.sum(X*X, axis=1, keepdims=True)
    Y_sum = jnp.sum(Y*Y, axis=1, keepdims=True)
    ones = jnp.ones((1, X_sum.shape[0]), dtype=jnp.float32)
    dist_matrix = X_sum@ones + (Y_sum@ones).T -2*(X@Y.T)
    dist_matrix = jnp.sqrt(jnp.abs(dist_matrix))
    return dist_matrix

@jit
def jax_gaussian_kernel_matrix(X, sigma):
    X_sum = jnp.sum(X*X, axis=1, keepdims=True)
    ones = jnp.ones((1, X_sum.shape[0]), dtype=jnp.float32)
    dist_matrix = X_sum@ones + (X_sum@ones).T -2*(X@X.T)

    affinity_matrix = jnp.exp(-0.5* (dist_matrix / (sigma**2)) )

    return affinity_matrix

def gaussian_kernel_matrix(X, sigma):
    X_sum = np.sum(X*X, axis=1, keepdims=True)
    ones = np.ones((1, X_sum.shape[0]), dtype=np.float32)
    dist_matrix = X_sum@ones + (X_sum@ones).T -2*(X@X.T)

    affinity_matrix = np.exp(-0.5* (dist_matrix / (sigma**2)) )

    return affinity_matrix

def calc_distance_matrix(features, feature_weights=None, metric='euclidean', use_jax=False):
    '''
    Args:
        features: np.array (matrix)
        feature_weights: np.array (vector)
            Weight assigned to each column
        metric: str
            It depends on scipy.spatial.distance.cdist
    Returns:
        dist_matrix: np.array (matrix)
    ref: https://github.com/palm-ml/valen/blob/master/utils/utils_graph.py
    '''

    # set weights
    if feature_weights is not None:
        features = features * np.tile(feature_weights, (features.shape[0], 1)) # Increase feature_weigths in the row direction and compute the element product
    
    # calc 
    if metric == 'euclidean':
        if use_jax:
            dist_matrix = jax_euclidean_dist_matrix(jnp.array(features), jnp.array(features))
            dist_matrix = np.array(dist_matrix)
        else:
            dist_matrix = distance.cdist(features, features, metric=metric)
    
    return dist_matrix

def calc_affinity_matrix(features, feature_weights=None, metric='gaussian', gaussian_sigma=1.0, use_jax=False):
    '''
    Args:
        features: np.array (matrix)
        feature_weights: np.array (vector)
            Weight assigned to each column
        metric: str
            It depends on scipy.spatial.distance.cdist
    Returns:
        dist_matrix: np.array (matrix)
    ref: https://github.com/palm-ml/valen/blob/master/utils/utils_graph.py
    '''
    # set weights
    if feature_weights is not None:
        features = features * np.tile(feature_weights, (features.shape[0], 1)) # Increase feature_weigths in the row direction and compute the element product
    
    # calc
    if metric == "gaussian":
        if use_jax:
            aff_matrix = jax_gaussian_kernel_matrix(jnp.array(features), sigma=gaussian_sigma)
            aff_matrix = np.array(aff_matrix)
        else:
            aff_matrix = gaussian_kernel_matrix(features, sigma=gaussian_sigma)
        
    return aff_matrix


def calc_adjecent_matrix(features, feature_weights=None, metric='euclidean', k=10, gaussian_sigma=1.0,
                         measure_time=False, use_jax=False, logger=None):
    '''
    Args:
        features: np.array (matrix)
        feature_weights: np.array (vector)
            Weight assigned to each column
        metric: str
            It depends on scipy.spatial.distance.cdist
        k: int
            the number of neighbours
    Returns:
        adj_matrix: np.array (matrix)
    ref: https://github.com/palm-ml/valen/blob/master/utils/utils_graph.py
    '''

    affinity = False # if True, adjecent matrix is created based on affinity matrix
    if metric in ['gaussian']:
        affinity = True
    
    N = features.shape[0] # sample size
    index_list = [i for i in range(0, N)]

    adj_matrix = np.zeros((N,N), dtype=np.float32)

    if affinity: # Calcucation based on affinity matrix
        if measure_time:    
            print("Calculation affinity matrix...", end="")
        time_s = time.time()
        aff_matrix = calc_affinity_matrix(features=features, feature_weights=feature_weights, metric=metric, 
                                          gaussian_sigma=gaussian_sigma, use_jax=use_jax)
        time_e = time.time()
        if measure_time:    
            print(f"    {time_e - time_s:.1f} [sec]")
            if logger is not None:
                logger.info(f"Calculation affinity matrix...    {time_e - time_s:.1f} [sec]")


        if measure_time:    
            print("Calculation adjecent matrix...", end="")
        time_s = time.time()

        # Assign the maximum value to the diagonal component so that the diagonal component is not chosen
        min_aff_val = np.min(aff_matrix) - 1
        aff_matrix[index_list, index_list] = min_aff_val

        # create adjecent matrix
        for _ in range(0, k):
            max_aff_list = np.argmax(aff_matrix, axis=1) # Get the index of the nearest neighbor for each sample
            aff_matrix[index_list, max_aff_list] = min_aff_val
            adj_matrix[index_list, max_aff_list] = 1

        time_e = time.time()
        if measure_time:    
            print(f"    {time_e - time_s:.1f} [sec]")
            if logger is not None:
                logger.info(f"Calculation adjecent matrix...    {time_e - time_s:.1f} [sec]")
    
    else: # Calculation based on dissimilarity matrix
        if measure_time:    
            print("Calculation distance matrix...", end="")
        time_s = time.time()
        dist_matrix = calc_distance_matrix(features=features, feature_weights=feature_weights, metric=metric, use_jax=use_jax)
        time_e = time.time()
        if measure_time:    
            print(f"    {time_e - time_s:.1f} [sec]")
            if logger is not None:
                logger.info(f"Calculation distance matrix...    {time_e - time_s:.1f} [sec]")

        if measure_time:    
            print("Calculation adjecent matrix...", end="")
        time_s = time.time()

        # Assign the maximum value to the diagonal component so that the diagonal component is not chosen
        max_dist_val = np.max(dist_matrix) + 1
        dist_matrix[index_list, index_list] = max_dist_val

        # create adjecent matrix
        for _ in range(0, k):
            min_dist_list = np.argmin(dist_matrix, axis=1) # Get the index of the nearest neighbor for each sample
            dist_matrix[index_list, min_dist_list] = max_dist_val
            adj_matrix[index_list, min_dist_list] = 1

        time_e = time.time()
        if measure_time:    
            print(f"    {time_e - time_s:.1f} [sec]")
            if logger is not None:
                logger.info(f"Calculation adjecent matrix...    {time_e - time_s:.1f} [sec]")


    return adj_matrix


def weight_optimization_qp(features, adj_matrix, feature_weights=None, use_jax=False, n_parallel=1):
    '''
    ref: Sun et al.2020 PP-PLL: Probability Propagation for Partial Label Learning  https://link.springer.com/chapter/10.1007/978-3-030-46147-8_8
    Args:
        features: np.ndarray (sample_size, the number of features)
            instances
        adj_matrix: np.ndarray (sample_size, sample_size)
            adjecent matrix of instances
        feature_weights: np.array (the number of features)
            Weight assigned to each column
        use_jax: bool
            when it is `True`, some calculations will be executed using JAX.
        n_parallel: int
            the number of parallel executions
        
    Returns:
        weight_matrix: np.ndarray (sample_size, sample_size)
    '''

    N = features.shape[0] # sample size
    weight_matrix = np.zeros((N, N), dtype=np.float32)
    param_size = adj_matrix[0].tolist().count(1) # 'k' -nearest neighbours

    # set weights
    if feature_weights is not None:
        features = features * np.tile(feature_weights, (features.shape[0], 1)) # Increase feature_weigths in the row direction and compute the element product
    
    
    if n_parallel >1:

        def objective_func(w_i, x_i, x_adj):
            return 0.5 * w_i.T @ (2*x_adj @ x_adj.T) @ w_i - 2 * x_i.T @ x_adj.T @ w_i


        def weight_opt_row(i):
            x_i = features[i]
            x_adj = features[adj_matrix[i]==1, :]

            result = minimize(objective_func, x0 = [0.0]*param_size, method='SLSQP', bounds=[(0, None)]*param_size, args=(x_i,x_adj),
                              constraints=({'type':'eq', 'fun': lambda w: np.sum(w)-1}))
            return result.x

        with ProcessPool(nodes=n_parallel) as pool:
            optimized_weights = pool.map(weight_opt_row, [i for i in range(N)])
            for i in range(N):
                weight_matrix[i, adj_matrix[i]==1] = optimized_weights[i]


    else:
        for i in range(N):
            x_i = features[i]
            x_adj = features[adj_matrix[i]==1, :]
            param_size = x_adj.shape[0]

            def objective_func(w):
                return 0.5 * w.T @ (2*x_adj @ x_adj.T) @ w - 2 * x_i.T @ x_adj.T @ w
            
            w0 = [0.0] * param_size
            if use_jax:
                optimizer = jaxopt.ScipyBoundedMinimize(fun=objective_func, method="SLSQP", jit=True)
                result = optimizer.run(init_params=jnp.array(w0), bounds=([0]*param_size, [jnp.inf]*param_size))
                weight_matrix[i, adj_matrix[i]==1] = np.array(result.params)
            else:
                result= minimize(objective_func, w0, method="SLSQP", bounds=[(0, None)]*param_size)
                weight_matrix[i, adj_matrix[i]==1] = result.x


    return weight_matrix



def normalize_matrix_old(matrix, use_jax=False):
    '''
    ref: https://github.com/palm-ml/valen/blob/master/utils/utils_graph.py
    Args:
        matrix : np.array
    '''
    if use_jax:
        rowsum = matrix.sum(1)
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.sparse.diags(r_inv).toarray()
        matrix = jax_matmul(jnp.array(r_mat_inv), jnp.array(matrix))
        matrix = np.array(matrix)

    else:
        rowsum = matrix.sum(1)
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.sparse.diags(r_inv).toarray()
        matrix = np.matmul(r_mat_inv, matrix)
    return matrix

def normalize_matrix(matrix, use_jax=False):
    normalized_matrix = matrix  / np.tile(matrix.sum(1).reshape(-1,1), (1, matrix.shape[1]))
    return normalized_matrix

def create_graph(features, feature_weights=None, 
                 knn_metric='euclidean', k=10, gaussian_sigma=1.0,
                 measure_time=False, use_jax=True, n_parallel=1, logger=None):
    '''
    creating a similarity graph of instances
    Args:
        features: np.array (sample_size, the number of features)
        feature_weights: list
            list of weights of features (columns)
        knn_metric: str
            choices: 'euclidean', 'gaussian'
        k: int
            This is the number of neighbours
        gaussian_sigma: float
            standard deviation of gaussian kernel
        measure_time: bool
            when it is `True`, the execution time of each process will be measured
        use_jax: bool
            when it is `True`, some calculations will be executed using JAX.
        n_parallel: int
            the number of parallel executions.

    Returns:
        adjecent_matrix: np.ndarray (sample_size, sample_size)
            adjecent matrix of the created similarity graph
        weight_matrix: np.ndarray (sample_size, sample_size)
            weight matrix of the created similarity graph
    '''
    
    if measure_time:    
        print("Creating graph...")

    assert k is not None

    # Create adjecent matrix (use only not complemented features)
    adjecent_matrix = calc_adjecent_matrix(features=features.copy(), 
                                            feature_weights=feature_weights, metric=knn_metric, k=k, 
                                            gaussian_sigma=gaussian_sigma,
                                            measure_time=measure_time, use_jax=use_jax, logger=logger)

    # Create weight matrix
    if measure_time:    
        print("weight optimization...", end="")
    # Weight (of graph directed edge) optimization 
    # ref: Sun et al.2020 PP-PLL: Probability Propagation for Partial Label Learning  https://link.springer.com/chapter/10.1007/978-3-030-46147-8_8
    time_s = time.time()
    weight_matrix = weight_optimization_qp(features=features.copy(), 
                                            adj_matrix=adjecent_matrix, 
                                            feature_weights=feature_weights,
                                            use_jax=False, n_parallel=n_parallel)
    weight_matrix = normalize_matrix(weight_matrix, use_jax=use_jax) # normalization

    time_e = time.time()
    if measure_time:  
        print(f"    {time_e - time_s:.1f} [sec]")  
        if logger is not None:
            logger.info(f"weight optimization ...    {time_e - time_s:.1f} [sec]")

        

    return adjecent_matrix, weight_matrix

@jit
def jax_matmul(A,B):
    return jnp.matmul(A, B)

def confident_propagation(comp_features, weight_matrix, comp_uniques, correct_own_comp, use_jax=False, init_comp_features=None):
    '''
    Args:
        comp_features: np.ndarray (sample_size, the number of CFs)
        weight_matrix: np.array (sample_size, sample_size)
        comp_uniques: list
            list of the number of unique values of each CF (This order is the same as the column order of comp_feature.)
        correct_own_comp: bool
            whether appling correct own complementary labels at each propagation or not
        use_jax: bool
            when it is `True`, some calculations will be executed using JAX.
    Returns:
        comp_features: np.ndarray (sample_size, the number of CFs)
    '''
    
    # Compute sharing value and Sharing complementary value
    # propagation
    if use_jax:
        sharing_vec = np.array(jax_matmul(jnp.array(weight_matrix), jnp.array(comp_features)))
    else:
        sharing_vec = np.matmul(weight_matrix, comp_features)

    # correct own complementary values
    if correct_own_comp:
        assert init_comp_features is not None
        comp_features = sharing_vec * np.where(init_comp_features==0.0, 0, 1) 
    else:
        comp_features = sharing_vec

    # normalization
    col_num_cumsum = 0
    for col_num in comp_uniques:
        l = col_num_cumsum
        r = col_num_cumsum + col_num
        comp_features[:, l:r] = comp_features[:, l:r] / np.tile(comp_features[:, l:r].sum(1).reshape(-1,1), (1, col_num)) # normalization each row
        col_num_cumsum += col_num


    return comp_features

def iterative_confident_propagation(comp_features, weight_matrix, comp_uniques,
                                    correct_own_comp,
                                    max_prop_iter=5,
                                    measure_time=False, use_jax=True, logger=None):
    '''
    Args:
        comp_features: np.ndarray (sample_size, the number of CFs)
        weight_matrix: np.array (sample_size, sample_size)
        comp_uniques: list
            list of the number of unique values of each CF (This order is the same as the column order of comp_feature.)
        correct_own_comp: bool
            whether appling correct own complementary labels at each propagation or not
        max_prop_iter: int
            the number of propagations
        prop_threshold: float
            stop threshold of propagating iteration
        use_jax: bool
            when it is `True`, some calculations will be executed using JAX.
    Returns:
        comp_features: np.ndarray (sample_size, the number of CFs)
    '''
    
    if measure_time:    
        print("Share complementary labels...", end="")
    time_s = time.time()

    init_comp_features = comp_features.copy()

    comp_features = confident_propagation(comp_features=comp_features,  
                                          comp_uniques=comp_uniques,
                                            weight_matrix=weight_matrix, 
                                            correct_own_comp=correct_own_comp, 
                                            use_jax=use_jax,
                                            init_comp_features=init_comp_features)
    
    for _ in range(1, max_prop_iter):
        old_comp_features = comp_features.copy()
        comp_features = confident_propagation(comp_features=comp_features, 
                                              comp_uniques=comp_uniques,
                                              weight_matrix=weight_matrix, 
                                              correct_own_comp=correct_own_comp, 
                                              use_jax=use_jax,
                                              init_comp_features=init_comp_features)



    time_e = time.time()
    if measure_time:  
        print(f"    {time_e - time_s:.1f} [sec]")  
        if logger is not None:
            logger.info(f"Share complementary labels...   {time_e - time_s:.1f} [sec]")

    return comp_features


def class_mass_normalization(comp_features, comp_prior_features, comp_uniques):
    '''
    ref: MLZhang, FYu. 2016 Solving the Partial Label Learning Problem: An Instance-based Approach
    Args:
        comp_features: np.ndarray (sample_size, the number of CFs (OneHot encoded))
        comp_prior_features: np.ndarray (sample_size, the number of CFs (OneHot encoded))
        comp_uniques: list
            list of the number of unique values of each CF (This order is the same as the column order of comp_feature.)
    Returns:
        output_features: np.ndarray (sample_size, the number of CFs)
    '''
    output_features = comp_features.copy()

    col_num_cumsum = 0
    for col_num in comp_uniques:
        l = col_num_cumsum
        r = col_num_cumsum + col_num

        normalization_factor = np.sum(comp_prior_features[:, l:r], axis=0) / np.sum(comp_features[:, l:r], axis=0)
        output_features[:, l:r] = np.tile(normalization_factor, (comp_features.shape[0], 1)) * comp_features[:, l:r]

        col_num_cumsum += col_num
    
    return output_features


def compute_feature_weight(df, onehot_names_list, comp_onehot_names_list, onehot_weights_dict,
                           round_alpha, knn_metric):
    '''
    computing feature weights
    Args:
        df: pd.DataFrame
        onehot_names_list: list
            list of one-hot encoded features' names in df
        comp_onehot_names_list: list
            list of one-hot encoded CFs' names in df
        onehot_weights_dict: dict
            key: column name, value: weight
        round_alpha: float
            choices: [0,1]
        knn_metric: str
            choices: 'euclidean', 'gaussian'
    Returns:
        np.array of feature_weights
    '''
    if knn_metric in ['euclidean', 'gaussian']:
        update_coef = np.sqrt(round_alpha)
    else:
        raise NotImplementedError
    
    feature_weights = []
    for col in df.columns.tolist():
        if col in onehot_names_list: # categorical feature
            if col in comp_onehot_names_list:
                feature_weights.append(onehot_weights_dict[col] * update_coef)
            else:
                feature_weights.append(onehot_weights_dict[col])
        else: # numerical or binary feature
            feature_weights.append(1.0)

    return np.array(feature_weights)


        

def propose_method(df, comp_cols, cat_cols=None, 
                   k=20, knn_metric='euclidean',
                   correct_own_comp=True,
                   gaussian_sigma=1.0, 
                   max_prop_iter=100, use_CMN=False,
                   round_alpha=1.0, round_max=1, round_threshold=1e-4, 
                   measure_time=False, use_jax=False, n_parallel=1, logger=None):
    '''
    Args:
        df: pd.DataFrame
            This dataframe includes data (index, target columns must not be included!)
        comp_cols: list of str
            List of complementary feature names
        cat_cols: list of str
            List of categorical feature names
        k: int
            This is the number of neighbours
        knn_metric: str
            choices: 'euclidean', 'gaussian'
        correct_own_cmp: bool
            whether appling correct own complementary labels at each propagation or not
        gaussian_sigma: float
            standard deviation of gaussian kernel
        max_prop_iter: int
            the number of propagations
        use_CMN: bool
            whether using CMN or not
        round_alpha: float
            the weight on the estimated exact value of CF when iteratively applying the proposed method
        round_max: int
            the number of iterative execution of the proposed method
        round_thershold: float
            stop threshold of iterative execution of the proposed method
        measure_time: bool
            when it is `True`, the execution time of each process will be measured
        use_jax: bool
            when it is `True`, some calculations will be executed using JAX.
        n_parallel: int
            the number of parallel executions.
            
    Returns:
        output_df: pd.DataFrame
            comp features have soft labels 
        output_CMN_df: pd.DataFrame
            comp features have hard labels
    '''

    # Preparation =====================================================================================================================

    output_df = df.copy()
    output_CMN_df = df.copy()
    N = df.shape[0] # sample size

    # About categorical cols
    onehot_names_dict = {} # key: original col name, val: col names (one-hot encoded)
    onehot_weights_dict = {} # key: one-hot encoded feature name, val: weight to compute distances
    onehot_names_list = [] # col names (one-hot encoded)
    comp_onehot_names_list = [] # complementary col names (one-hot encoded)
    for cat_col in cat_cols:
        onehot_col_names = [col for col in df.columns.tolist() if cat_col in col]
        
        onehot_names_dict[cat_col] = onehot_col_names
        onehot_names_list += onehot_col_names
        if cat_col in comp_cols:
            comp_onehot_names_list += onehot_col_names
        
        for onehot_col in onehot_col_names:
            if knn_metric in ['euclidean', 'gaussian']:
                onehot_weights_dict[onehot_col] = 1.0 / np.sqrt(len(onehot_col_names))
            else:
                raise NotImplementedError
 

    # Set one-hot vector's weights to compute distances
    feature_weights = []
    for col in df.columns.tolist():
        if col in onehot_names_list: # categorical feature
            feature_weights.append(onehot_weights_dict[col])
        else: # numerical feature
            feature_weights.append(1.0)



    # Create graph ====================================================================================================================
    # Compute feature weights
    feature_weights = compute_feature_weight(df=df.drop(comp_onehot_names_list, axis=1), 
                                            onehot_names_list=onehot_names_list,
                                            comp_onehot_names_list=comp_onehot_names_list,
                                            onehot_weights_dict=onehot_weights_dict,
                                            round_alpha=1.0, knn_metric=knn_metric)
    adjecent_matrix, weight_matrix = create_graph(features=df.drop(comp_onehot_names_list, axis=1).values,
                                                  feature_weights=feature_weights,
                                                  knn_metric=knn_metric, 
                                                  k=k, gaussian_sigma=gaussian_sigma,
                                                  measure_time=measure_time, use_jax=use_jax, n_parallel=n_parallel, logger=logger)
    
    
    # Share complementary labels ===================================================================================================
    comp_uniques = []
    for col in comp_cols: # comp_onehot_names_list orders of comp_cols and comp_onehot_names_list are same from code creating comp_onehot_names_list
        comp_uniques.append(len(onehot_names_dict[col]))
    comp_features = iterative_confident_propagation(comp_features=df.loc[:, comp_onehot_names_list].values, 
                                                    weight_matrix=weight_matrix,
                                                    comp_uniques=comp_uniques,
                                                    correct_own_comp=correct_own_comp,
                                                    max_prop_iter=max_prop_iter,
                                                    measure_time=measure_time, use_jax=use_jax, logger=logger)
    
            
    output_df.loc[:, comp_onehot_names_list] = comp_features # Shaing

    if (round_max < 1) or (round_alpha == 0.0):
        if use_CMN:
            # class mass normalization
            comp_CMN_features = class_mass_normalization(comp_features=comp_features, 
                                                    comp_prior_features=df.loc[:, comp_onehot_names_list].values,
                                                    comp_uniques=comp_uniques,
                                                    )
            output_CMN_df.loc[:, comp_onehot_names_list] = comp_CMN_features 

            return output_df, output_CMN_df
        else:
            return output_df
    
    # Iterative updating ===========================================================================================================
    if measure_time:    print("Start Iterative updating...")

    
    for iter_idx in range(round_max):
        if measure_time:    print(f"{iter_idx+1}th iter start...")

        # Compute feature weights
        feature_weights = compute_feature_weight(df=output_df, 
                                                 onehot_names_list=onehot_names_list,
                                                 comp_onehot_names_list=comp_onehot_names_list,
                                                 onehot_weights_dict=onehot_weights_dict,
                                                 round_alpha=round_alpha, knn_metric=knn_metric)

        # Create graph
        adjecent_matrix, weight_matrix = create_graph(features=output_df.values,
                                                  feature_weights=feature_weights,
                                                  knn_metric=knn_metric, 
                                                  k=k, gaussian_sigma=gaussian_sigma,
                                                  measure_time=measure_time, use_jax=use_jax, n_parallel=n_parallel, logger=logger)


        # Share complementary labels
        comp_features = iterative_confident_propagation(comp_features=df.loc[:, comp_onehot_names_list].values, 
                                                        weight_matrix=weight_matrix,
                                                        comp_uniques=comp_uniques,
                                                        correct_own_comp=correct_own_comp,
                                                        max_prop_iter=max_prop_iter,
                                                        measure_time=measure_time, use_jax=use_jax, logger=logger)

        # decide stop or not
        stop_flag =False
        if np.max(np.abs(comp_features - output_df.loc[:, comp_onehot_names_list].values)) <= round_threshold:
            stop_flag = True 

        # sharing
        output_df.loc[:, comp_onehot_names_list] = comp_features

        if stop_flag:
            break

    if use_CMN:
        # class mass normalization
        comp_CMN_features = class_mass_normalization(comp_features=output_df.loc[:, comp_onehot_names_list].values, 
                                                    comp_prior_features=df.loc[:, comp_onehot_names_list].values, 
                                                    comp_uniques=comp_uniques,
                                                    )
        output_CMN_df.loc[:, comp_onehot_names_list] = comp_CMN_features
        
        return output_df, output_CMN_df

    else:
        return output_df
    
# existing method ============================================================================================================================================
    
# IPAL
from scipy.optimize import minimize

class IPAL():
    def __init__(self, k, alpha, T, use_jax=True, n_parallel=1):
        '''
        Args:
            k: int
                This is the number of nearest neighbours considered
            alpha: float
                the balancing coefficient in (0,1)
            T: int
                the number of iterations
        '''
        self.k = k
        self.alpha = alpha
        self.T = T
        self.use_jax = use_jax
        self.n_parallel = n_parallel
        
        self.have_data = False
        self.weight_matrix = None
        self.initial_confidences = None
    
    def regist_data(self, instances, instance_weights=None):
        '''
        Args:
            instances: np.array (matrix) (the number of samples, the number of columns)
            instance_weights: np.array (vector)
                Weight assigned to each column

        '''
        self.instances = instances.copy()
        if instance_weights is None:
            self.instance_weights = None
        else:
            self.instance_weights = instance_weights.copy()

        self.have_data = True

    def create_graph(self, instances=None, instance_weights=None):
        '''
        Args:
            instances: np.array (matrix) (the number of samples, the number of columns)
            instance_weights: np.array (vector)
                Weight assigned to each column

        '''
        if instances is None:
            if self.have_data == False:
                print("You should set 'feature'")
                exit()
        else:
            self.regist_data(instances=instances, instance_weights=instance_weights)
                
        self.adjacent_matrix = calc_adjecent_matrix(features=self.instances.copy(), 
                                               feature_weights=self.instance_weights, 
                                               metric='euclidean', 
                                               k=self.k, 
                                               use_jax=self.use_jax)
        
        self.weight_matrix = weight_optimization_qp(features=self.instances.copy(),
                                               adj_matrix=self.adjacent_matrix,
                                               feature_weights=self.instance_weights,
                                               use_jax=False,
                                               n_parallel=self.n_parallel)
        self.weight_matrix = normalize_matrix(self.weight_matrix, use_jax=self.use_jax)

    def _confidence_propagation_IPAL(self, confidences):

        assert self.weight_matrix is not None
        assert self.initial_confidences is not None

        return self.alpha * np.matmul(self.weight_matrix, confidences)  + (1.0 - self.alpha) * self.initial_confidences

    def _class_mass_normalization(self, target_confidences):
        
        assert self.initial_confidences is not None

        normalization_factor = np.sum(self.initial_confidences, axis=0) / np.sum(target_confidences, axis=0)
        return np.tile(normalization_factor, (target_confidences.shape[0], 1)) * target_confidences

    def predict_transductive(self, target_confidences, return_conf=False):
        '''
        Args:
            targets_confidences: np.array (the number of data, the number of unique values)
                one-hot encoded target feature     
        '''

        assert self.weight_matrix.shape[0] == target_confidences.shape[0] # require the number of instances is equivalent with the number of target samples

        self.initial_confidences = target_confidences.copy()
        self.target_confidences = target_confidences.copy()

        # propagation
        for _ in range(self.T):
            self.target_confidences = self._confidence_propagation_IPAL(self.target_confidences)
            self.target_confidences = self.target_confidences / np.tile(self.target_confidences.sum(1).reshape(-1,1), (1, self.target_confidences.shape[1])) # normalization each row
        # CMN
        CMN_target_confidences = self._class_mass_normalization(self.target_confidences)
        predictive_labels = np.argmax(CMN_target_confidences, axis=1) # label encoded e.g. 0, 1, 2, ...
        n_labels = CMN_target_confidences.shape[1] # because (n_samples, n_uniques)
        predictive_ohe_labels = np.eye(n_labels)[predictive_labels]

        self.predictive_labels = predictive_labels.copy()
        self.predictive_ohe_labels = predictive_ohe_labels.copy()

        if return_conf:
            return predictive_ohe_labels, self.target_confidences
        else:
            return predictive_ohe_labels

    
    def predict_inductive(self, new_instances, feature_weights=None):
        '''
        Args:
            new_instances: np.array (n_samples, n_features)
        '''
        
        def objective_func(w_i, x_i, x_adj):
            return 0.5 * w_i.T @ (2*x_adj @ x_adj.T) @ w_i - 2 * x_i.T @ x_adj.T @ w_i

        # set weights
        if feature_weights is not None:
            new_instances = new_instances * np.tile(feature_weights, (new_instances.shape[0], 1)) # Increase feature_weigths in the row direction and compute the element product
        
        new_predictive_labels = np.zeros(new_instances.shape[0], dtype=np.int32)


        if self.n_parallel > 1:

            def decide_label_row(i):
                # calculate dist vector new_instances[i] <-> self.instances
                dist_vector = np.sqrt(np.sum( (self.instances - new_instances[i])**2, axis=1)) # euclidean distance
                # the indices of k neaest neighbors about new_instances[i]
                KNN_indices = np.argsort(dist_vector)[:self.k]

                neighboring_instances = self.instances[KNN_indices, :]
                neighboring_labels = self.predictive_labels[KNN_indices]

                x_i   = new_instances[i]
                x_adj = neighboring_instances
                result = minimize(objective_func, x0=[0.0] * self.k, method='SLSQP', bounds=[(0, None)]*self.k, args=(x_i,x_adj),
                                constraints=({'type':'eq', 'fun': lambda w: np.sum(w)-1}) )
                w_star = result.x

                min_error = 1e10
                min_label = -1
                for l in np.unique(neighboring_labels):
                    tmp_error = np.sqrt( np.sum( (new_instances[i] - w_star[np.where(neighboring_labels == l)] @ neighboring_instances[np.where(neighboring_labels == l)])**2  ))
                    if tmp_error <= min_error:
                        min_error = tmp_error
                        min_label = l
                return min_label
            
            with ProcessPool(nodes=self.n_parallel) as pool:
                new_predictive_labels = pool.map(decide_label_row, [i for i in range(len(new_instances))])


        else:

            for i in range(len(new_instances)):
                # calculate dist vector new_instances[i] <-> self.instances
                dist_vector = np.sqrt(np.sum( (self.instances - new_instances[i])**2, axis=1)) # euclidean distance

                # the indices of k neaest neighbors about new_instances[i]
                KNN_indices = np.argsort(dist_vector)[:self.k]

                neighboring_instances = self.instances[KNN_indices, :]
                neighboring_labels = self.predictive_labels[KNN_indices]


                x_i   = new_instances[i]
                x_adj = neighboring_instances
                result = minimize(objective_func, x0=[0.0] * self.k, method='SLSQP', bounds=[(0, None)]*self.k, args=(x_i,x_adj),
                                constraints=({'type':'eq', 'fun': lambda w: np.sum(w)-1}) )
                w_star = result.x

                min_error = 1e10
                min_label = -1
                for l in np.unique(neighboring_labels):
                    tmp_error = np.sqrt( np.sum( (new_instances[i] - w_star[np.where(neighboring_labels == l)] @ neighboring_instances[np.where(neighboring_labels == l)])**2  ))
                    if tmp_error <= min_error:
                        min_error = tmp_error
                        min_label = l
                new_predictive_labels[i] = min_label
        
        new_predictive_ohe_labels = np.eye(self.predictive_ohe_labels.shape[1])[new_predictive_labels]
        
        return new_predictive_ohe_labels