# Evaluating Large Language Models on Formal Mathematics

First create a conda environment using environment.yml

`conda create --name environment_name --file environment.yml`

## Building Updated IsarStep

`cd build_isarstep`

### Replicating Results

Follow the instructions in the README of Build\_Train\_IsarStep to build the training set.
Follow the instructions in the README of Build\_Validation\_IsarStep to build the validation set.

The above folders are pulled from [here](https://github.com/Wenda302/Build_IsarStep).
These scripts were written by Wenda Li and modified by Jeffrey Ma to produce the desired splits.

### Alternative Splits

To make an alternative split of the current AFP, just alter train\_names.txt.
All the names that appear in train\_names.txt will be processed and built into the training dataset and all excluded names will be built into the validation set.
There are helper functions in ../scripts/data/ to collect theory names and count lines of code.

## Fine-tuning Models

`cd scripts`

Create a new config yml in configs/ based on config.yml.
Make sure finetune.py uses the config file you want.

`python finetune.py`

## Evaluating Models

Make sure gen.py and eval.py use the correct config files.
Note that n specifies how many answers to produce for each question.
If n > 1, only top-n accuracy will be evaluated.
If n = 1, top-1 accuracy and BLEU score will be evaluated.

To generate model output run:
`python gen.py`

To evaluate the output run:
`python eval.py`

To visualize attention weights, use viz.ipynb and plugin your model.

## Results

The 2019 IsarStep, 2022 IsarStep, and small IsarStep are available in datasets/
The trained models are available under models/ 
The outputs of the models are available under outputs/

## Data Analysis

Scripts used for processing and analyzing the dataset are under scripts/data/
Some of the scripts were run on Yale's HPC Grace cluster due to the size of the datasets.
The scripts to run these slurm jobs on the cluster are found under grace/
