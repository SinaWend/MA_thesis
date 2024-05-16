#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-0:00:00
#SBATCH --qos=cpu_normal
#SBATCH --partition=cpu_p
#SBATCH --output=runs_output/BRCA-dinov2-Washington-bs32-gpu-20epochs-erm-%j.out
#SBATCH --error=runs_output/BRCA-dinov2-Washington-bs32-gpu-20epochs-erm-%j.err

python main_out.py -c ./examples/conf/TCGA_BRCA.yaml
