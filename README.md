# Learning from Complementary Features

## requirements

* Python: 3.9.14

* Library: Please see `requirements.txt`

## Directory Structure

```
learning_from_cl/
　├README.md
　├config/
　│　├IPAL_adult_compColAll_AEC3_base.yaml
　│　├IPAL_adult_compColAll_base.yaml
　│　├IPAL_bank_compColAll_base.yaml
　│　├adult_knn_compColAll_AEC3_base.yaml
　│　├adult_knn_compColAll_base.yaml
　│　├bank_knn_compColAll_base.yaml
　│　└sample.yaml
　├data/
　│　├adult/
　│　│　├ ...
　│　└bank/
　│　│　├ ...
　├libs/
　│　├evaluation.py
　│　├helpers.py
　│　├learning.py
　│　├load_data.py
　│　├methods.py
　│　├utils.py
　│　└utils_processing.py
　├requirements.txt
　├main.py
　├exp_adult.sh
　└test.sh
```

* `config`: This is a directory that stores YAML files, which contain arguments that are common and fixed across the experiment scripts.
* `data`: This is a directory that stores the datasets used in the experiments. Each data must be downloaded from [UCI repository](https://archive.ics.uci.edu/).
* `libs`: This is a directory that stores functions and other utilities used in main.py.
* `main.py`: This is the experiment script.
* `exp_bank.sh`: This is a shell script for conducting experiments with the Bank dataset. Running this script as-is will take a considerable amount of time, so please adjust the hyperparameter settings for the experiment as needed before executing it.
* `test.sh`: This script is used to check whether the experimental program runs.

## Datasets download Links

* [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
  * Unzip `bank+marketing.zip`, and place all the files inside `bank.zip` into `learning_from_cl/data/bank/`.

* [Adult](https://archive.ics.uci.edu/dataset/2/adult)
  * Unzip `adult.zip`, and place all the files located directly under it into `learning_from_cl/data/adult/`.

## How to Execute Experiments

```bash

# sample script (very short time)
python main.py --config_file ./config/sample.yaml

# full experiments using Bank dataset (very long time)
bash ./exp_bank.sh

```

The explanation of the main arguments is follow:

**Experiental Settings:**
* `dataset_name`: Using dataset name. Please select from ['bank', 'adult'].

* `main_dir`: The path of `learning_from_cl` directory.

* `data_dir`: The path of `data` directory.

* `output_dir`: Path of output directory for log data. 

* `sample_size`: The total number of data to be used. If set to a negative value, all data will be used.

* `test_rate`: The proportion of test data among the data used.

* `comp_cols`: This parameter specifies which categorical features (discrete variables with more than three values) to treat as CFs. If set to all, all categorical features will be treated as CFs.

* `n_parallel`: The number of parallel executions.

* `seed`: The random seed value.

* `measure_time`: When it is `True`, the execution time of each process will be measured.

* `use_jax`: When it is `True`, some calculations will be executed using JAX.

* `method`: this parameter specifies the method to be used. Choose from [`ord`, `comp`, `propose`, `IPAL`].
  * `ord`:  When the exact value of CF is used.
  * `comp`: When the observed value of CF is used as is.
  * `propose`: The proposed method in this paper.
  * `IPAL`: Zhang and Yu, 2015

**Proposed Method Settings:**

* `knn_metric`: This parameter specifies the distance metric to be used in KNN. In this paper, `euclidean` was used.

* `k`: This parameter specifies the value of the hyperparameter 'k' for the proposed method, which represents the number of neighbors in KNN.

* `round_alpha`: This paramter specifies the value of the hyperparameter 'gamma' for the proposed method, corresponding to the weight on the estimated exact value of CF when iteratively applying the proposed method.

* `max_prop_iter`: This parameter specifies the value of the hyperparameter 'T' for the proposed method, which represents the number of confidence propagation iterations.

* `correct_own_comp`: When this is `True`, the process described in Eq. (24) of this paper will be executed in the proposed method.

**IPAL Settings:**

* `k`: This parameter specifies the value of the hyperparameter 'k' for the proposed method, which represents the number of neighbors in KNN.
* `max_prop_iter`: This parameter specifies the value of the hyperparameter 'T' for the proposed method, which represents the number of confidence propagation iterations.
* `ipal_alpha`: This parameter specifies the value of the hyperparameter 'alpha' for IPAL [Zhang and Yu, 2015]．

## References

[Zhang and Yu, 2015] M.-L. Zhang and F. Yu, “Solving the partial label learning problem: An
instance-based approach.” in IJCAI, 2015, pp. 4048–4054.
