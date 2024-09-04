#!/bin/bash

#SBATCH --gpus=2
#SBATCH --mem=64G
#SBATCH --time=2-0:00:00
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --output=../runs_output/TCGAnew-vitlarge-Harvard-bs16-erm-%j.out
#SBATCH --error=../runs_output/TCGAnew-vitlarge-Harvard-bs16-erm-%j.err

python main_out.py -c ../conf/TCGA_tissue.yaml
