#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-0:00:00
#SBATCH --qos=cpu_normal
#SBATCH --partition=cpu_p
#SBATCH --output=./runs_output/download_camelyon17_wilds-%j.out
#SBATCH --error=./runs_output/download_camelyon17_wilds-%j.err

module load python/3.x  # Load the appropriate Python module

python download_camelyon17.py
