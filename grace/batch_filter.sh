#!/bin/bash

#SBATCH --job-name=isarstep_filter
#SBATCH --time=1-
#SBATCH --cpus-per-task=48
#SBATCH --mail-user=jeffrey.ma.jzm5@yale.edu
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_output_viz.out

module load miniconda

source activate myenv

python filter_dataset.py extracted_isar_dataset/CausalSteps/source.txt extracted_isar_dataset/CausalSteps/target.txt 