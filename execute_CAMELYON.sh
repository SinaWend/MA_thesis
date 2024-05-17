#!/bin/bash

#SBATCH --gpus=2
#SBATCH --mem=64G
#SBATCH --time=2-0:00:00
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --output=../runs_output/CAMELYON-dinov2s-center4-bs32-20epochs-gpu-erm-%j.out
#SBATCH --error=../runs_output/CAMELYON-dinov2s-center4-bs32-20epochs-gpu-erm-%j.err

python main_out.py -c ../conf/CAMELYON.yaml
