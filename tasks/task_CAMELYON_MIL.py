import sys
import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from domainlab.tasks.task_dset import mk_task_dset
from domainlab.tasks.utils_task import ImSize

def extract_domain_number(domain_str):
    # Extract the numerical part from the domain string (e.g., 'center0' -> 0)
    return int(domain_str[-1])

def one_hot_encode(labels, num_classes):
    return torch.nn.functional.one_hot(labels, num_classes=num_classes)

class HistopathologyDataset(Dataset):
    def __init__(self, image_paths, label, domain, transform=None, num_classes=4):
        self.image_paths = image_paths
        self.label = torch.tensor(label, dtype=torch.long)
        self.domain = domain
        self.transform = transform
        self.num_classes = num_classes
        print(f"Created HistopathologyDataset with {len(self.image_paths)} images, label {self.label}, and domain {self.domain}")

    def __len__(self):
        return 1  # Each instance is a bag containing multiple images

    def __getitem__(self, idx):
        images = []
        for p in self.image_paths:
            try:
                img = Image.open(p).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {p}: {e}")
        
        if not images:
            raise ValueError("No images found in the bag.")

        # Stack images to create a single batch-like tensor
        images_tensor = torch.stack(images)
        label_one_hot = one_hot_encode(self.label, self.num_classes)
        
        domain_num = extract_domain_number(self.domain)
        domain_tensor = torch.tensor(domain_num, dtype=torch.long)
        domain_one_hot = one_hot_encode(domain_tensor, 5)
        return images_tensor, label_one_hot, domain_one_hot


def get_task(na=None):
    dim_y = 4  # Adjust as per your number of classes
    task = mk_task_dset(isize=ImSize(3, 224, 224), dim_y=dim_y, taskna="wsi_stage_prediction")

    base_path = '/lustre/groups/shared/histology_data/CAMELYON17/patches/2.0'
    csv_path = '/lustre/groups/shared/histology_data/CAMELYON17/stage_labels.csv'  # Update with the correct path
    img_trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # Load the stage information from the CSV file
    df_labels = pd.read_csv(csv_path)
    print("Loaded CSV file")
    
    # Map stages to integer labels
    stage_mapping = {'negative': 0, 'micro': 1, 'macro': 2, 'itc': 3}  # Extend this as needed for other stages
    df_labels['stage'] = df_labels['stage'].map(stage_mapping)
    print("Mapped stages to integers")

    # Use WSI file names (excluding .zip) and stages for each WSI folder
    df_labels = df_labels[df_labels['patient'].str.endswith('.tif')]
    stage_dict = dict(zip(df_labels['patient'].str.replace('.tif', '', regex=False), df_labels['stage']))
    print(f"Stage dictionary: {stage_dict}")

    domain_dict = {i: f'center{i // 20}' for i in range(100)}  # Patient ID to center mapping
    print("Created domain dictionary")

    domain_datasets_train = {}
    domain_datasets_val = {}
    sample = True
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder}")
            patient_id = folder.split('_')[1]  # Extract patient ID from folder name
            domain_name = domain_dict[int(patient_id)]
            print(f"Domain Name: {domain_name}")

            # Collect .png images from the folder
            image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
            print(f"Image Paths: {image_paths}")
            
            # Get the stage label for this WSI folder
            wsi_label_key = folder  # folder name should match the key in stage_dict
            stage_label = stage_dict.get(wsi_label_key, None)
            print(f"Stage Label: {stage_label}")

            if stage_label is None or not image_paths:
                print(f"Skipping folder {folder} due to missing stage label or images.")
                continue
            if sample:
            # Sample patches from the collected images
                dataframe = pd.DataFrame({'path': image_paths, 'label': [stage_label] * len(image_paths)})
                print(f"Dataframe before sampling: {dataframe}")
                
                sampled_df = dataframe.groupby('label').apply(lambda x: x.sample(n=5, random_state=42)).reset_index(drop=True)
                image_paths = sampled_df['path'].tolist()
                print(f"Sampled Image Paths: {image_paths}")

            dataset = HistopathologyDataset(image_paths, stage_label, domain_name, transform=img_trans, num_classes=dim_y)

            if domain_name not in domain_datasets_train:
                domain_datasets_train[domain_name] = []
                domain_datasets_val[domain_name] = []

            if int(patient_id) % 4 == 0:  # Every fourth patient goes into the validation set
                domain_datasets_val[domain_name].append(dataset)
            else:
                domain_datasets_train[domain_name].append(dataset)

    for domain_name in domain_datasets_train:
        combined_dataset_train = ConcatDataset(domain_datasets_train[domain_name])
        combined_dataset_val = ConcatDataset(domain_datasets_val[domain_name])
        print(f"Adding domain {domain_name} with {len(domain_datasets_train[domain_name])} training datasets and {len(domain_datasets_val[domain_name])} validation datasets")
        task.add_domain(name=domain_name, dset_tr=combined_dataset_train, dset_val=combined_dataset_val)

    return task


