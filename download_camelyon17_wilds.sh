#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-0:00:00
#SBATCH --qos=cpu_normal
#SBATCH --partition=cpu_p
#SBATCH --output=./runs_output/camelyon17_download-%j.out
#SBATCH --error=./runs_output/camelyon17_download-%j.err

# Load necessary modules (if any)
module load wget
module load unzip

# Define the URL and target directory
URL="https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/"
TARGET_DIR="/lustre/groups/shared/histology_data/CAMELYON17_WILD"
ZIP_FILE="$TARGET_DIR/camelyon17.zip"

# Create target directory if it doesn't exist
mkdir -p $TARGET_DIR

# Download the dataset
echo "Downloading the dataset..."
wget -O $ZIP_FILE $URL

# Check if download was successful
if [[ $? -ne 0 ]]; then
    echo "Download failed!"
    exit 1
fi

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

echo "Download and extraction completed successfully."
