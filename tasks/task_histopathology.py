from domainlab.tasks.task_dset import mk_task_dset
import os
import json
import torch
from PIL import Image
from torchvision import transforms
from domainlab.tasks.utils_task import ImSize
from tasks.patches_processing import process_slides_primary_diagnosis_test
from torch.utils.data import ConcatDataset



class HistopathologyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None, num_classes=3):
        self.dataframe = dataframe
        self.transform = transform
        self.num_classes = num_classes
        self.targets = torch.tensor(self.dataframe['label'].values, dtype=torch.long)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']  # Access path
        label = self.dataframe.iloc[idx]['label']  # Access corresponding label
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        label_one_hot = one_hot_encode(label, self.num_classes)
        

        return image, label_one_hot


def one_hot_encode(labels, num_classes):
    return torch.nn.functional.one_hot(labels, num_classes=num_classes)
###########


def get_task(na=None):
    # Initialize the task
    dim_y = 4
    task = mk_task_dset(isize=ImSize(3, 224, 224), dim_y=dim_y, taskna="custom_histopathology_task")
    diagnosis_to_label = {}

    # Define transformations
    img_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Initialize a dictionary to hold lists of datasets by domain
    domain_datasets_train = {}
    domain_datasets_val = {}

    # Loop through each folder in the TCGA directory
    for folder in os.listdir("data/TCGA"):
        folder_path = os.path.join("data/TCGA", folder)
        if os.path.isdir(folder_path):
            # Process each 'biospecimen' file to determine domain_name
            for file in os.listdir(folder_path):
                if file.startswith("biospecimen") and file.endswith(".json"):
                    json_path = os.path.join(folder_path, file)
                    with open(json_path) as json_file:
                        biospecimen_data_list = json.load(json_file)
                        if biospecimen_data_list and isinstance(biospecimen_data_list, list):
                            domain_name = biospecimen_data_list[0]["samples"][0]["portions"][0]["analytes"][0]["aliquots"][0]["center"]["name"]

            # Process slides to get data
            slide_path = folder_path
            save_path = os.path.join(folder_path, "output_test")
            coords, data = process_slides_primary_diagnosis_test(slide_path, save_path, diagnosis_to_label)

            if len(data) > 10:
                data = data.sample(n=10).reset_index(drop=True)  # Sample 10 patches if data is large

            # Split dataset
            df_train, df_val, df_test = split_dataset(data, 0.7, 0.3)

            # Create datasets
            dataset_train = HistopathologyDataset(df_train, transform=img_trans, num_classes=dim_y)
            dataset_val = HistopathologyDataset(df_val, transform=img_trans, num_classes=dim_y)

            # Accumulate datasets by domain, creating a list if key doesn't exist
            if domain_name not in domain_datasets_train:
                domain_datasets_train[domain_name] = []
                domain_datasets_val[domain_name] = []
            domain_datasets_train[domain_name].append(dataset_train)
            domain_datasets_val[domain_name].append(dataset_val)

    # Combine datasets for each domain and add to the task
    for domain_name in domain_datasets_train:
        combined_dataset_train = ConcatDataset(domain_datasets_train[domain_name])
        combined_dataset_val = ConcatDataset(domain_datasets_val[domain_name])
        task.add_domain(name=domain_name, dset_tr=combined_dataset_train, dset_val=combined_dataset_val)

    print(diagnosis_to_label)
    return task




def split_dataset(df, train_split, val_split):
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    
    # Calculate split indices
    idx_train_end = int(len(df_shuffled) * train_split)
    idx_val_end = int(len(df_shuffled) * (train_split + val_split))
    
    # Split the dataframe
    df_train = df_shuffled.iloc[:idx_train_end]
    df_val = df_shuffled.iloc[idx_train_end:idx_val_end]
    df_test = df_shuffled.iloc[idx_val_end:]
    
    return df_train, df_val, df_test