from domainlab.tasks.task_dset import mk_task_dset
import os
import json
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
from PIL import Image
from torchvision import transforms
from domainlab.tasks.utils_task import ImSize
from examples.tasks.patches_processing import process_slides
from torch.utils.data import ConcatDataset
import pandas as pd



class HistopathologyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None, num_classes=2):
        self.dataframe = dataframe
        #print(self.dataframe)
        self.transform = transform
        self.num_classes = num_classes
        self.targets = torch.tensor(self.dataframe['label'].values.astype(int), dtype=torch.long)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']  # Access path
        label = int(self.dataframe.iloc[idx]['label'])  # Access corresponding label, ensure it's integer
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        label_one_hot = one_hot_encode(label, self.num_classes)
        return image, label_one_hot

def one_hot_encode(labels, num_classes):
    return torch.nn.functional.one_hot(labels, num_classes=num_classes)


def get_task(na=None):
    dim_y = 2
    task = mk_task_dset(isize=ImSize(3, 224, 224), dim_y=dim_y, taskna="custom_histopathology_task")

    # Augmentations for training
    img_trans_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    # Basic transformations for validation and testing
    img_trans_val_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    domain_datasets_train = {}
    domain_datasets_val = {}
    annotations_path = '/lustre/groups/shared/histology_data/CAMELYON17/annotations'
    centers_paths = [
        '/lustre/groups/shared/histology_data/CAMELYON17/slides/center0',
        '/lustre/groups/shared/histology_data/CAMELYON17/slides/center1',
        '/lustre/groups/shared/histology_data/CAMELYON17/slides/center2',
        '/lustre/groups/shared/histology_data/CAMELYON17/slides/center3',
        '/lustre/groups/shared/histology_data/CAMELYON17/slides/center4',
    ]

    def extract_info(filename):
        parts = filename.split('_')
        patient_id = parts[1]
        node_id = parts[3][0]
        return patient_id, node_id

    save_path = '/lustre/groups/aih/sina.wendrich/MA_code/output_CAMELYON17'

    slides_processing = False
    if slides_processing:
        for filename in os.listdir(annotations_path):
            if filename.endswith('.xml'):
                patient_id, node_id = extract_info(filename)
                slide_filename = f'patient_{patient_id}_node_{node_id}.tif'
                for center_path in centers_paths:
                    slide_path = os.path.join(center_path, slide_filename)
                    if os.path.exists(slide_path):
                        folder_path = center_path
                        annotation_path = os.path.join(annotations_path, filename)
                        coords, data = process_slides(slide_path, annotation_path, save_path, center_path)
                        df_train, df_val, df_test = split_dataset(data, 0.7, 0.3)
                        dataset_train = HistopathologyDataset(df_train, transform=img_trans_train, num_classes=dim_y)
                        dataset_val = HistopathologyDataset(df_val, transform=img_trans_val_test, num_classes=dim_y)
                        domain_name = os.path.basename(center_path)
                        if domain_name not in domain_datasets_train:
                            domain_datasets_train[domain_name] = []
                            domain_datasets_val[domain_name] = []
                        domain_datasets_train[domain_name].append(dataset_train)
                        domain_datasets_val[domain_name].append(dataset_val)
                        break
    else:
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
            dataset_train = HistopathologyDataset(df_center_train, transform=img_trans_train, num_classes=dim_y)
            dataset_val = HistopathologyDataset(df_center_val, transform=img_trans_val_test, num_classes=dim_y)
            update_datasets(domain_datasets_train, domain_datasets_val, dataset_train, dataset_val, center_name)

    for domain_name in domain_datasets_train:
        combined_dataset_train = ConcatDataset(domain_datasets_train[domain_name])
        combined_dataset_val = ConcatDataset(domain_datasets_val[domain_name])
        task.add_domain(name=domain_name, dset_tr=combined_dataset_train, dset_val=combined_dataset_val)

    return task

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
