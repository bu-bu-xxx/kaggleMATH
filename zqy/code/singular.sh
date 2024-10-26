#!/bin/bash

#SBATCH --partition=h07gpuq1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30000M
#SBATCH -J "bubu-kaggle"
#SBATCH --gres=gpu:1

SCRIPT_PATH="/puhome/24112456g/kaggleMATH/zqy/code/singular_qwen.py"

python3 $(SCRIPT_PATH)


