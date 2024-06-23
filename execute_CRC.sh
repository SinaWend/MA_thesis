#!/bin/bash

#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=2-0:00:00
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --output=runs_output/CRC-vitbase-Baylor-bs16-erm-%j.out
#SBATCH --error=runs_output/CRC-vitbase-Baylor-bs16-erm-%j.err

python DomainLab/main_out.py -c conf/CRC_mil.yaml
