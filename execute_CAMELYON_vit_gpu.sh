#!/bin/bash

#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=2-0:00:00
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --output=../runs_output/CAMELYON-vitlarge-dann0.001-center4-bs32-20epochs-%j.out
#SBATCH --error=../runs_output/CAMELYON-vitlarge-dann0.001-center4-bs32-20epochs-%j.err

python main_out.py -c ../conf/CAMELYON_vit.yaml
