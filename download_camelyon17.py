import os
from wilds import get_dataset

# Specify the directory where you want to save the dataset
save_dir = "/lustre/groups/shared/histology_data/CAMELYON17_WILD"

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Get the Camelyon17 dataset
dataset = get_dataset(dataset="camelyon17", download=True, root_dir=save_dir)

print("Camelyon17 dataset downloaded successfully!")
