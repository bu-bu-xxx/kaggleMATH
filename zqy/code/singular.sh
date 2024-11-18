#!/bin/bash

#SBATCH --partition=h07gpuq1
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30000M
#SBATCH -J "bubu-kaggle"
#SBATCH --gres=gpu:1

PATH="/scratch/bbl/kaggleMATH/"
SCRIPT_PATH="/scratch/bbl/kaggleMATH/zqy/code/singular_qwen.py"

module load singularity/3.11.3

singularity exec \
	--bind /scratch/bbl/kaggleMATH/:/mnt/kaggleMATH \
	--nv \
	/scratch/bbl/singlarity_images/python_transformers.sif \
	python /mnt/kaggleMATH/zqy/code/singular_qwen.py



















