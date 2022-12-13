#!/bin/bash

#SBATCH --job-name=isarstep_visualize
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mail-user=jeffrey.ma.jzm5@yale.edu
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_output_viz.out

module load miniconda

source activate myenv

python dataset_visualization.py postcheckpoint_isar/CausalSteps/