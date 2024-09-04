#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-0:00:00
#SBATCH --qos=cpu_normal
#SBATCH --partition=cpu_p
#SBATCH --output=./runs_output/camelyon17_extract-%j.out
#SBATCH --error=./runs_output/camelyon17_extract-%j.err

# Load necessary modules (if any)
module load unzip

# Define the zip file and target directory
ZIP_FILE="/lustre/groups/shared/histology_data/CAMELYON17_WILD/camelyon17.zip"
TARGET_DIR="/lustre/groups/shared/histology_data/CAMELYON17_WILD"

# Extract the dataset
echo "Extracting the dataset..."
unzip $ZIP_FILE -d $TARGET_DIR

# Check if extraction was successful
if [[ $? -ne 0 ]]; then
    echo "Extraction failed!"
    exit 1
fi

# Clean up
echo "Cleaning up..."
rm $ZIP_FILE

echo "Extraction completed successfully."
