#!/bin/bash

#SBATCH --gpus=2
#SBATCH --mem=64G
#SBATCH --time=2-0:00:00
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --output=runs_output/blood2-resnet-bs32-%j.out
#SBATCH --error=runs_output/blood2-resnet-bs32-%j.err

python main_out.py -c ./examples/conf/vit_blood2.yaml

