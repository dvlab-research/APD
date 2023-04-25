#!/bin/sh

## uncomment for slurm
##SBATCH -p gpu
##SBATCH --gres=gpu:4
##SBATCH -c 20

export PYTHONPATH=./
#eval "$(conda shell.bash hook)"
#conda activate pt140  # pytorch 1.4.0 env
PYTHON=python3

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp tool/train.py tool/train.sh tool/test.sh tool/test.py ${config} ${exp_dir}

export PYTHONPATH=./
#$PYTHON -u ${exp_dir}/train.py \
#  --config=${config} \
#  2>&1 | tee ${model_dir}/train-$now.log

$PYTHON -u ${exp_dir}/test.py \
  --config=${config} \
  2>&1 | tee ${result_dir}/test-$now.log
