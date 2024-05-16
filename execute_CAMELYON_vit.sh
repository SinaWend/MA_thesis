#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-0:00:00
#SBATCH --qos=cpu_normal
#SBATCH --partition=cpu_p
#SBATCH --output=runs_output/vitlarge-center4-20epochs-erm-%j.out
#SBATCH --error=runs_output/vitlarge-center4-20epochs-erm-%j.err

python main_out.py -c ./examples/conf/CAMELYON_vit.yaml
