from argparse import ArgumentParser

from load_data import *

def arg_parser():
    parser = ArgumentParser()

    parser.add_argument("--config_file", help="config file name")

    parser.add_argument("--exp_name")
    parser.add_argument("--dataset_name", choices=['bank', 'adult'])
    parser.add_argument("--main_dir", default="./")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--method", choices=['ord', 'comp', 'propose', 'IPAL'])
    
    parser.add_argument("--knn_metric", choices=['euclidean', 'gaussian'], help="This parameter specifies the distance metric to be used in KNN. In this paper, `euclidean` was used. Please select from ['euclidean', 'gaussian']") 
    parser.add_argument("--k", type=int, help="This parameter specifies the value of the hyperparameter 'k' for the proposed method, which represents the number of neighbors in KNN.")
    parser.add_argument("--correct_own_comp", type=bool, help='When this is `True`, the process described in Eq. (24) of this paper will be executed in the proposed method.')
    parser.add_argument("--max_prop_iter", type=int, help="This parameter specifies the value of the hyperparameter 'T' for the proposed method, which represents the number of confidence propagation iterations.") 
    parser.add_argument("--prop_threshold", type=float, default=0.0001, help="stop threshold of propagating iteration")
    
    parser.add_argument("--round_max", type=int, default=1)    # use round_max=1 in this paper
    parser.add_argument("--round_alpha", type=float, help="This paramter specifies the value of the hyperparameter 'gamma' for the proposed method, corresponding to the weight on the estimated exact value of CF when iteratively applying the proposed method.") 

    parser.add_argument("--sample_size", type=int, help="The total number of data to be used. If set to a negative value, all data will be used.The total number of data to be used. If set to a negative value, all data will be used.")
    parser.add_argument("--test_rate", type=float, help="The proportion of test data among the data used. Please select from (0, 1)") 
    parser.add_argument("--comp_cols", type=str, nargs="+", default=['all'], help="list of features to be CFs") 
    parser.add_argument("--avoid_estimate_cols", type=str, nargs="+", default=[], help="This parameter specifies which categorical features (discrete variables with more than three values) to treat as CFs. If set to all, all categorical features will be treated as CFs.")
    
    parser.add_argument("--n_parallel", type=int, help="the number of using cores")
    parser.add_argument("--measure_time", type=bool, help="whether measuring each execution times or not")
    parser.add_argument("--use_jax", type=bool, default=True, help="whether using JAX or not")
    parser.add_argument("--seed", type=int, default=42, help="The random seed value")

    # for IPAL
    parser.add_argument("--ipal_alpha", type=float, help="alpha in IPAL")
    ## k: -> k
    ## T: -> max_prop_iter

    return parser

def get_args_default():
    '''
    get a dict has default args' values
    '''
    default_dict = {
        'main_dir': '../',
        'data_dir': '../../../opt/nas/data',
        'output_dir': '../output',

        'k': 20,

        'correct_own_comp': True,

        'max_prop_iter': 100,
        'prop_threshold': 0.0001,

        'round_max': 1,
        'round_alpha': 0.2,

        'sample_size': -1,
        'test_rate': 0.5,
        'comp_cols': ['all'],
        'avoid_estimate_cols':[],

        'n_parallel': 4,
        'measure_time': True,
        'use_jax': True,
        'seed': 42,

        # for IPAL
        'ipal_alpha' : 1.0,
        
    }

    return default_dict

def get_log_filename(args):
    '''
    get log file name
    '''
    name = ""

    name += args.dataset_name
    name += '_' + args.method

    # hyperparameters of methods
    if args.method in ['propose']:

        name += '_KM' + args.knn_metric
        name += '_k' + str(args.k)

        if args.correct_own_comp:
            name += '_cocTrue'
        else:
            name += '_cocFalse'
        
        name += '_PI' + str(args.max_prop_iter)
        name += '_PT' + str(args.prop_threshold)

        name += '_RM' + str(args.round_max)
        if args.round_max > 0:
            name += '_RA' + str(args.round_alpha)

    elif args.method == 'IPAL':
        name += '_k' + str(args.k)
        name += '_alpha' + str(args.ipal_alpha)
        name += '_T' + str(args.max_prop_iter)
    

    # dataset settings
    name += '_size' + str(args.sample_size)
    name += '_test'+ str(args.test_rate)

    if args.method in ['comp', 'propose', 'IPAL']:
        if args.comp_cols == ['all']:
            name += '_CColAll'
        else:
            name += '_CCol' + str(comp_cols_code(args.dataset_name, args.comp_cols))
    
    if args.method in ['propose', 'IPAL']:
        if len(args.avoid_estimate_cols) >= 1:
            name += '_AEC' + str(comp_cols_code(args.dataset_name, args.avoid_estimate_cols))

    
    # computing settings
    name += '_para' + str(args.n_parallel)
    name += '_seed' + str(args.seed)

    return name


def pseudo_args(args_dict):
    class Args():
        tmp = "ttt"
    args = Args()
    for k, v in args_dict.items():
        if v is not None:
            setattr(args, k, v)
    return args
