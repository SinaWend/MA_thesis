from domainlab.tasks.task_dset import mk_task_dset
import os
import json
import torch
from PIL import Image
from torchvision import transforms
from domainlab.tasks.utils_task import ImSize
from torch.utils.data import ConcatDataset
import pandas as pd


class HistopathologyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None, cancer_transform=None, num_classes=2, oversample_factor=1):
        self.dataframe = dataframe
        self.transform = transform
        self.cancer_transform = cancer_transform
        self.num_classes = num_classes
        self.adjusted_indices = []
        for i, label in enumerate(dataframe['label']):
            self.adjusted_indices.extend([i] * (oversample_factor if label == 1 else 1))

    def __len__(self):
        return len(self.adjusted_indices)

    def __getitem__(self, idx):
        actual_idx = self.adjusted_indices[idx]
        img_path = self.dataframe.iloc[actual_idx]['path']
        label = int(self.dataframe.iloc[actual_idx]['label'])
        domain = self.dataframe.iloc[actual_idx]['center']  # Extract domain
        image = Image.open(img_path).convert("RGB")
        if label == 1 and self.cancer_transform:
            image = self.cancer_transform(image)
        elif self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        label_one_hot = one_hot_encode(label_tensor, self.num_classes)
        # print(domain)
        domain_num = extract_domain_number(domain)
        domain_tensor = torch.tensor(domain_num, dtype=torch.long)
        domain_one_hot = one_hot_encode(domain_tensor, 5)
        return image, label_one_hot, domain_one_hot  # Return domain here

def extract_domain_number(domain_str):
    # Extract the numerical part from the domain string (e.g., 'center0' -> 0)
    return int(domain_str[-1])

def one_hot_encode(labels, num_classes):
    return torch.nn.functional.one_hot(labels, num_classes=num_classes)


# Define the different transformations
def get_transforms():
    img_trans_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
    ])

    img_cancer_augment_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # More aggressive cropping
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),  # Increased rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        ], p=0.5),
        transforms.ToTensor(),
    ])

    img_trans_val_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return img_trans_train, img_cancer_augment_train, img_trans_val_test

def get_task(na=None):
    dim_y = 2
    task = mk_task_dset(isize=ImSize(3, 224, 224), dim_y=dim_y, taskna="custom_histopathology_task")
    img_trans_train, img_cancer_augment_train, img_trans_val_test = get_transforms()

    domain_datasets_train = {}
    domain_datasets_val = {}
 

    save_path = '/lustre/groups/aih/sina.wendrich/MA_code/output_CAMELYON17'
    
    patches_path = os.path.join(save_path, "patches/8")
    data = []
    for slide_dir in os.listdir(patches_path):
        slide_path = os.path.join(patches_path, slide_dir)
        for filename in os.listdir(slide_path):
            if filename.endswith('.png'):
                parts = filename.split('_')
                label = int(parts[-1].split('.')[0])
                center_name = parts[-2]
                data.append({
                    'path': os.path.join(slide_path, filename),
                    'label': label,
                    'center': center_name
                })

    dataframe = pd.DataFrame(data)
    df_train, df_val, df_test = split_dataset(dataframe, 0.7, 0.3)
    for center_name in set(dataframe['center']):
        df_center_train = df_train[df_train['center'] == center_name]
        df_center_val = df_val[df_val['center'] == center_name]
        if center_name == 'center0':
            transform = img_trans_val_test
            cancer_transform = img_trans_val_test
            oversample_factor = 1
        else:
            transform = img_trans_train
            cancer_transform = img_cancer_augment_train
            oversample_factor = 30

        # Balance each center's training data considering the oversampling
        df_center_train_balanced = balance_classes(df_center_train, oversample_factor, 'center0')

        dataset_train = HistopathologyDataset(df_center_train_balanced, transform=transform, cancer_transform=cancer_transform, num_classes=dim_y, oversample_factor=oversample_factor)
        dataset_val = HistopathologyDataset(df_center_val, transform=img_trans_val_test, num_classes=dim_y)
                
        update_datasets(domain_datasets_train, domain_datasets_val, dataset_train, dataset_val, center_name)

    for domain_name in domain_datasets_train:
        combined_dataset_train = ConcatDataset(domain_datasets_train[domain_name])
        combined_dataset_val = ConcatDataset(domain_datasets_val[domain_name])
        task.add_domain(name=domain_name, dset_tr=combined_dataset_train, dset_val=combined_dataset_val)

    return task

def balance_classes(dataframe, oversample_factor=1, exclude_center=None):
    # Only balance classes if the center is not the one to exclude
    if dataframe['center'].iloc[0] == exclude_center:
        return dataframe  # Return the original dataframe if it's the excluded center
    
    cancer_df = dataframe[dataframe['label'] == 1]
    noncancer_df = dataframe[dataframe['label'] == 0]
    
    effective_cancer_count = len(cancer_df) * oversample_factor
    if effective_cancer_count < len(noncancer_df):
        noncancer_df = noncancer_df.sample(effective_cancer_count, random_state=42)
    else:
        cancer_needed = (len(noncancer_df) + oversample_factor - 1) // oversample_factor
        if cancer_needed < len(cancer_df):
            cancer_df = cancer_df.sample(cancer_needed, random_state=42)
    
    return pd.concat([cancer_df, noncancer_df])


def split_dataset(df, train_split, val_split):
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    idx_train_end = int(len(df_shuffled) * train_split)
    idx_val_end = int(len(df_shuffled) * (train_split + val_split))
    df_train = df_shuffled.iloc[:idx_train_end]
    df_val = df_shuffled.iloc[idx_train_end:idx_val_end]
    df_test = df_shuffled.iloc[idx_val_end:]
    return df_train, df_val, df_test

def update_datasets(domain_datasets_train, domain_datasets_val, dataset_train, dataset_val, center_name):
    domain_name = os.path.basename(center_name)
    if domain_name not in domain_datasets_train:
        domain_datasets_train[domain_name] = []
        domain_datasets_val[domain_name] = []
    domain_datasets_train[domain_name].append(dataset_train)
    domain_datasets_val[domain_name].append(dataset_val)
