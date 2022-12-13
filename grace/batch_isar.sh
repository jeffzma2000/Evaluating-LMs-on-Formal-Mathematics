#!/bin/bash

#SBATCH --job-name=isarstep_train_dataset
#SBATCH --partition=week
#SBATCH --time=3-12:00:00
#SBATCH --cpus-per-task=36
#SBATCH --mail-user=jeffrey.ma.jzm5@yale.edu
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_output.out

module load miniconda

source activate myenv

python Build_IsarStep/build_isarstep_database.py --isa_bin Isabelle2022_modified/bin/isabelle --isar_data isar_dataset/
