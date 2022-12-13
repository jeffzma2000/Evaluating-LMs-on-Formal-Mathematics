#!/bin/bash

#SBATCH --job-name=isarstep_train_dataset_final_step
#SBATCH --partition=week
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=36
#SBATCH --mail-user=jeffrey.ma.jzm5@yale.edu
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_output_more.out

module load miniconda

source activate myenv

python Build_IsarStep/extract_isarstep_from_database.py --dir_in isar_dataset --processed_id 202212210369