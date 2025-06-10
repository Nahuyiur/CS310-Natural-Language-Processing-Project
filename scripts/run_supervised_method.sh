#!/bin/bash
#SBATCH -o xlm-roberta-base-en-mix.%j.log
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH -J bert-base-chinese-zh-mix
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate nlp

nvidia-smi

python /lab/haoq_lab/cse12310520/NLP/proj/supervised_method/save_best_model.py