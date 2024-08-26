import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, log_loss)
import itertools 

def evaluation_classification(true_value, pred_label, pred_prob=None, n_classes=2):
    '''
    evaluation predicting results for classification
    Args:
        true_value: np.array (sample_size, 1)
        pred_label: np.array (sample_size, 1)
        pred_prob: np.array (sample_size, the number of uniq values)
        n_classes: n_classes of the classification problem
    Return:
        scores: dict (key: score_name, val: score_value)
    '''

    label_scores = [
        accuracy_score,
        f1_score,
        precision_score,
        recall_score
    ]

    prob_scores = [
        roc_auc_score,
        log_loss
    ]

    scores = {}

    for f in label_scores:
        if n_classes == 2: # binary classification
            score = f(y_true=true_value, y_pred=pred_label)
        else: # multi-class classification
            if f != accuracy_score:
                score = f(y_true=true_value, y_pred=pred_label, average="macro")
            else:
                score = f(y_true=true_value, y_pred=pred_label)
        scores[str(f.__name__)] = score

    if pred_prob is not None:
        _normalization_term = np.sum(pred_prob, axis=1)
        _normalization_term = np.repeat(_normalization_term[None, :], n_classes, axis=0).T
        pred_prob = pred_prob / _normalization_term

        for f in prob_scores:
            if n_classes == 2: # binary classification
                if f == roc_auc_score:
                    score = f(y_true=true_value, y_score=pred_prob[:,1])
                elif f == log_loss:
                    score = f(y_true=true_value, y_pred=pred_prob, labels=[i for i in range(n_classes)])
            else: # multi-class classification
                if f == roc_auc_score:
                    score = f(y_true=true_value, y_score=pred_prob, average="macro", multi_class="ovr")
                elif f == log_loss:
                    score = f(y_true=true_value, y_pred=pred_prob, labels=[i for i in range(n_classes)])
            scores[str(f.__name__)] = score

    return scores


def average_cross_entropy(df_true, df_pred, comp_cols):
    '''
    calculation of average cross entropy of CFs' estimated values
    Args:
        df_true: pd.DataFrame includes CFs' exact values (these values are represented by OneHot vectors, and each column name must includes the feature name) 
        df_pred: pd.DataFrame includes CFs' estimated values (these values are represented by OneHot vectors, and each column name must includes the feature name)
        comp_cols: list of CF names
    Returns:
        score: score averaged over scores from multiple features
        score_dict: dict (key: feature name, val: score)
    '''
    score = 0
    score_dict = {}

    # ref: https://masaki-note.com/2022/04/29/cross-entropy-error/
    def np_log(x):
        return np.log(np.clip(a=x, a_min=1e-10, a_max=1e+10))

    for col in comp_cols:
        col_onehot_list = [c for c in df_true.columns.tolist() if col in c]
        true_vec = df_true.loc[:, col_onehot_list].values
        pred_vec = df_pred.loc[:, col_onehot_list].values
        
        col_score = (-true_vec * np_log(pred_vec)).sum(axis=1).mean()
        score += col_score
        score_dict[col] = col_score


    return score, score_dict

def average_entropy(df, comp_cols):
    '''
    calculation of average entropy of CFs' any values
    Args:
        df pd.DataFrame includes CFs' any values (these values are represented by OneHot vectors, and each column name must includes the feature name)
        comp_cols: list of CF names
    Returns:
        score: score averaged over scores from multiple features
        score_dict: dict (key: feature name, val: score)
    '''
    score = 0
    score_dict = {}

    # ref: https://masaki-note.com/2022/04/29/cross-entropy-error/
    def np_log(x):
        return np.log(np.clip(a=x, a_min=1e-10, a_max=1e+10))

    for col in comp_cols:
        col_onehot_list = [c for c in df.columns.tolist() if col in c]
        prob_vec = df.loc[:, col_onehot_list].values

        col_score = (-prob_vec * np_log(prob_vec)).sum(axis=1).mean()
        score+= col_score
        score_dict[col] = col_score
    
    return score, score_dict


def entropy(prob_vec):
    # ref: https://masaki-note.com/2022/04/29/cross-entropy-error/
    def np_log(x):
        return np.log(np.clip(a=x, a_min=1e-10, a_max=1e+10))
    
    return (-prob_vec * np_log(prob_vec)).sum(axis=1).mean()

def cross_entropy(y_true_vec, y_pred_vec):
    # ref: https://masaki-note.com/2022/04/29/cross-entropy-error/
    def np_log(x):
        return np.log(np.clip(a=x, a_min=1e-10, a_max=1e+10))
    
    return (-y_true_vec * np_log(y_pred_vec)).sum(axis=1).mean()


def evaluation_pred_prob(true_value, pred_prob, n_classes):
    '''
    evaluation prediction probabilities for classification
    Args:
        true_value: np.array (sample_size, 1)
        pred_prob: np.array (sample_size, the number of uniq values)
        n_classes: n_classes of the classification problem
    Return:
        scores: dict (key: score_name, val: score_value)
    '''

    prob_scores = [
        roc_auc_score,
        log_loss,
    ]

    scores = {}

    for f in prob_scores:
        if n_classes == 2: # binary classification
            if f == roc_auc_score:
                score = f(y_true=true_value, y_score=pred_prob[:,1])
            elif f == log_loss:
                score =f(y_true=true_value, y_pred=pred_prob[:,1])
            scores[str(f.__name__)] = score

        else: # multi-class classification
            if f == roc_auc_score:
                for average, multi_class in [('macro','ovr'), ('macro', 'ovo')]:
                    score = f(y_true=true_value, y_score=pred_prob, average=average, multi_class=multi_class)
                    scores[str(f.__name__) + f"_{average}_{multi_class}"] = score

            elif f in [log_loss]:
                score = f(y_true=true_value, y_pred=pred_prob)
                scores[str(f.__name__)] = score

    return scores

def evaluation_pred_prob_ohe(true_vec, pred_prob_vec):
    '''
    evaluation prediction probabilities for classification when true_values are represented by OneHot vectors.
    Args:
        true_value: np.array (sample_size, 1)
        pred_prob: np.array (sample_size, the number of uniq values)
        n_classes: n_classes of the classification problem
    Return:
        scores: dict (key: score_name, val: score_value)
    '''
    prob_ohe_scores = [
        cross_entropy,
        entropy
    ]

    scores = {}
    for f in prob_ohe_scores:
        if f == cross_entropy:
            score = f(y_true_vec=true_vec, y_pred_vec=pred_prob_vec)
        elif f == entropy:
            score = f(prob_vec=pred_prob_vec)
        
        scores[str(f.__name__)] = score
    
    return scores

def evaluation_pred_label(true_value, pred_value, n_classes):

    '''
    evaluation prediction probabilities for classification when each true_value is represented by a single value.
    Args:
        true_value: np.array (sample_size, 1)
        pred_prob: np.array (sample_size, the number of uniq values)
        n_classes: n_classes of the classification problem
    Return:
        scores: dict (key: score_name, val: score_value)
    '''
    label_scores = [
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    ]

    scores = {}

    for f in label_scores:
        if n_classes == 2: # binary classification
            score = f(y_true=true_value, y_pred=pred_value)
            scores[str(f.__name__)] = score
        else: # multi-class classification
            if f in [f1_score, precision_score, recall_score]:
                for average in ['micro', 'macro']:
                    score = f(y_true=true_value, y_pred=pred_value, average=average)
                    scores[str(f.__name__) + f"_{average}"] = score
            elif f in [accuracy_score, confusion_matrix]:
                score = f(y_true=true_value, y_pred=pred_value)
                scores[str(f.__name__)] = score
            else:
                raise NotImplementedError

    return scores

def evaluation_classification_detail(true_value, pred_label, pred_prob=None, n_classes=None):
    '''
    evaluation predicting results for classification
    Args:
        true_value: np.array (sample_size, 1)
        pred_label: np.array (sample_size, 1)
        pred_prob: np.array (sample_size, the number of uniq values)
        n_classes: n_classes of the classification problem
    Return:
        scores: dict (key: score_name, val: score_value)
    '''
    assert n_classes is not None

    scores = evaluation_pred_label(true_value=true_value, pred_value=pred_label, n_classes=n_classes)
    if pred_prob is not None:
        scores_prob = evaluation_pred_prob(true_value=true_value, pred_prob=pred_prob, n_classes=n_classes)
        scores.update(scores_prob)
    
    return scores
    

def evaluation_disamb_cls(df_true, df_pred_prob=None, df_pred_label=None, comp_cols=None, labeling_strategy='random'):
    '''
    evaluation estimating CF's exact values
    Args:
        df_true: pd.DataFrame includes CF's exact values
        df_pred_prob: pd.DataFrame includes CF's estimated values 
        pred_prob: pd.DataFrame includes estimated confidences
        comp_cols: list of CFs' feature names
        labeling_strategy: How to determine a single estimate when the estimation method is "comp" (choices: "random")
    Return:
        scores_per_col: dict (key: feature name, val: score name) of dict (key: score name, val: score value)
        scores_average: score averaged over scores from multiple features
    '''

    assert comp_cols is not None
    assert (df_pred_prob is not None) or (df_pred_label is not None)

    scores_per_col = {}
    scores_average = {}

    for idx, col in enumerate(comp_cols):
        col_onehot_list = [c for c in df_true.columns.tolist() if col in c]
        true_vec = np.argmax(df_true.loc[:, col_onehot_list].values, axis=1).astype(np.int32)
        true_ohe_vec = df_true.loc[:, col_onehot_list].values.astype(np.int32)

        if df_pred_prob is not None:
            pred_prob_vec = df_pred_prob.loc[:, col_onehot_list].values

        if df_pred_label is not None:
            pred_label_vec = np.argmax(df_pred_label.loc[:, col_onehot_list].values, axis=1)
        else:
            # labeling based pred_prob
            if labeling_strategy == 'random':
                pred_label_vec = []
                for i in range(pred_prob_vec.shape[0]):
                    hard_idx = np.random.choice(pred_prob_vec.shape[1], size=1, p=pred_prob_vec[i, :]/np.sum(pred_prob_vec[i, :]))
                    pred_label_vec.append(hard_idx)
                pred_label_vec = np.array(pred_label_vec)
            else:
                raise NotImplementedError
                
        col_scores_label = evaluation_pred_label(true_value=true_vec, pred_value=pred_label_vec, n_classes=len(col_onehot_list))
        
        if df_pred_prob is not None:
            col_scores_prob_ohe = evaluation_pred_prob_ohe(true_vec=true_ohe_vec, pred_prob_vec=pred_prob_vec)

        if df_pred_prob is not None:
            col_scores = dict(col_scores_label.copy(), **col_scores_prob_ohe)
        else:
            col_scores = col_scores_label.copy()
        
        scores_per_col[col] = col_scores.copy()

        if idx == 0:
            scores_average = col_scores.copy()
        else:
            for key in col_scores.keys():
                if key != 'confusion_matrix':
                    scores_average[key] += col_scores[key]
    
    for key in scores_average.keys():
        scores_average[key] = scores_average[key] / len(comp_cols)
    
    return scores_per_col, scores_average








