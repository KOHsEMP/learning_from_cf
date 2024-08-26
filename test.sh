#!/bin/bash

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


base_name='sample'

config_dir='config'
base_yaml="${config_dir}/${base_name}.yaml"


# propose
method="propose"

max_prop_iter=100
correct_own_comp=True
iter_alpha_list=(0.0)
k_list=(20)
round_alpha_list=(0.5)

n_parallel=4
seed_list=(42)

for seed in ${seed_list[@]}
do
    for k in ${k_list[@]}
    do
        for iter_alpha in ${iter_alpha_list[@]}
        do
            for round_alpha in ${round_alpha_list[@]}
            do
                tmp_yaml="${config_dir}/tmp_${base_name}_${method}_${knn_metric}_${weight_metric}_${k}_${gaussian_sigma}_${round_alpha}_${seed}.yaml"
                cp ${base_yaml} ${tmp_yaml}
                echo "" >> ${tmp_yaml}
                echo "method: ${method}" >> ${tmp_yaml}
                echo "k: ${k}" >> ${tmp_yaml}
                echo "max_prop_iter: ${max_prop_iter}" >> ${tmp_yaml}
                echo "correct_own_comp: ${correct_own_comp}" >> ${tmp_yaml}
                echo "iter_alpha: ${iter_alpha}" >> ${tmp_yaml}
                echo "n_parallel: ${n_parallel}" >> ${tmp_yaml}
                echo "round_alpha: ${round_alpha}" >> ${tmp_yaml}
                echo "seed: ${seed}" >> ${tmp_yaml}

                python main.py --config_file ${tmp_yaml}
                rm ${tmp_yaml}
            done
        done
    done
done