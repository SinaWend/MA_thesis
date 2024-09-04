from domainlab.tasks.task_dset import mk_task_dset
import os
import torch
from PIL import Image
from torchvision import transforms
from domainlab.tasks.utils_task import ImSize
from torch.utils.data import ConcatDataset
import pandas as pd


class HistopathologyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None, num_classes=2):
        self.dataframe = dataframe
        self.transform = transform
        self.num_classes = num_classes
        print(f"Dataset initialized with {len(self.dataframe)} samples.")
      
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']
        label = int(self.dataframe.iloc[idx]['label'])
        domain = self.dataframe.iloc[idx]['center']  # Extract domain
        #print(f"Loading image from {img_path}")
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            # You can return a default image or handle the error appropriately
            image = Image.new('RGB', (224, 224), (255, 255, 255))  # Create a blank white image
            label = 0  # Assign a default label
            domain = 0  # Assign a default domain
        if self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        label_one_hot = one_hot_encode(label_tensor, self.num_classes)
        domain_num = extract_domain_number(domain)
        domain_tensor = torch.tensor(domain_num, dtype=torch.long)
        domain_one_hot = one_hot_encode(domain_tensor, 5)
        print(f"Image loaded. Label: {label}, Domain: {domain}")
        return image, label_one_hot, domain_one_hot  # Return domain here

def extract_domain_number(domain_str):
    return int(domain_str)

def one_hot_encode(labels, num_classes):
    return torch.nn.functional.one_hot(labels, num_classes=num_classes)


# Define the different transformations
def get_transforms():
    img_trans_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
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
    print("Transformations defined.")
    return img_trans_train, img_trans_val_test

def get_task(na=None):
    dim_y = 2
    task = mk_task_dset(isize=ImSize(3, 224, 224), dim_y=dim_y, taskna="custom_histopathology_task")
    img_trans_train, img_trans_val_test = get_transforms()

    domain_datasets_train = {}
    domain_datasets_val = {}
    
    csv_path = '/lustre/groups/shared/histology_data/CAMELYON17_WILD/camelyon17_v1.0/metadata.csv'
    patches_path = '/lustre/groups/shared/histology_data/CAMELYON17_WILD/camelyon17_v1.0/patches'
    
    # Load the CSV
    print(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    print(f"CSV loaded. Number of rows: {len(df)}")
    
    # Build the dataframe with full image paths
    df['path'] = df.apply(lambda row: os.path.join(patches_path, f"patient_{int(row['patient']):03d}_node_{row['node']}", f"patch_patient_{int(row['patient']):03d}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png"), axis=1)
    df.rename(columns={'tumor': 'label'}, inplace=True)
    
    print("Dataframe after processing:")
    print(df.head())

    df_train, df_val, df_test = split_dataset(df, 0.7, 0.3)
    print(f"Train set size: {len(df_train)}, Validation set size: {len(df_val)}, Test set size: {len(df_test)}")
    
    for center_name in set(df['center']):
        df_center_train = df_train[df_train['center'] == center_name]
        df_center_val = df_val[df_val['center'] == center_name]
        print(f"Processing center: {center_name}, Train samples: {len(df_center_train)}, Val samples: {len(df_center_val)}")
        if center_name == 4:  # Assuming center 0 uses validation/test transforms
            transform = img_trans_val_test
        else:
            transform = img_trans_train
        dataset_train = HistopathologyDataset(df_center_train, transform=transform, num_classes=dim_y)
        dataset_val = HistopathologyDataset(df_center_val, transform=img_trans_val_test, num_classes=dim_y)
                
        update_datasets(domain_datasets_train, domain_datasets_val, dataset_train, dataset_val, center_name)

    for domain_name in domain_datasets_train:
        print(f"Combining datasets for domain: {domain_name}")
        combined_dataset_train = ConcatDataset(domain_datasets_train[domain_name])
        combined_dataset_val = ConcatDataset(domain_datasets_val[domain_name])
        task.add_domain(name=domain_name, dset_tr=combined_dataset_train, dset_val=combined_dataset_val)
        print(f"Domain {domain_name} added to the task with {len(combined_dataset_train)} training samples and {len(combined_dataset_val)} validation samples.")

    return task


def split_dataset(df, train_split, val_split):
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    idx_train_end = int(len(df_shuffled) * train_split)
    idx_val_end = int(len(df_shuffled) * (train_split + val_split))
    df_train = df_shuffled.iloc[:idx_train_end]
    df_val = df_shuffled.iloc[idx_train_end:idx_val_end]
    df_test = df_shuffled.iloc[idx_val_end:]
    print(f"Dataset split: {len(df_train)} train, {len(df_val)} validation, {len(df_test)} test samples.")
    return df_train, df_val, df_test

def update_datasets(domain_datasets_train, domain_datasets_val, dataset_train, dataset_val, center_name):
    domain_name = str(center_name)
    if domain_name not in domain_datasets_train:
        domain_datasets_train[domain_name] = []
        domain_datasets_val[domain_name] = []
    domain_datasets_train[domain_name].append(dataset_train)
    domain_datasets_val[domain_name].append(dataset_val)
    print(f"Datasets updated for domain: {domain_name}. Train size: {len(domain_datasets_train[domain_name])}, Val size: {len(domain_datasets_val[domain_name])}")
