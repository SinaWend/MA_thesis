import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from domainlab.tasks.task_dset import mk_task_dset
from domainlab.tasks.utils_task import ImSize
from tasks.patches_processing import process_slides_tissue_type

#class HistopathologyDataset(Dataset):
    # def __init__(self, image_paths, label, transform=None):
    #     self.image_paths = image_paths
    #     self.label = torch.tensor(label, dtype=torch.long)
    #     self.transform = transform

    # def __len__(self):
    #     return 1  # Each instance is a bag containing multiple images

    # def __getitem__(self, idx):
    #     images = []
    #     for p in self.image_paths:
    #         try:
    #             img = Image.open(p).convert('RGB')
    #             if self.transform:
    #                 img = self.transform(img)
    #             images.append(img)
    #         except FileNotFoundError:
    #             print(f"Error: File not found at {p}")
    #             continue  # or raise an exception if you prefer
    #     if not images:
    #         raise ValueError("No valid images in the bag.")
#     #     return torch.stack(images), self.label
# class HistopathologyDataset(Dataset):
#     def __init__(self, image_paths, label, transform=None):
#         self.image_paths = image_paths
#         self.label = torch.tensor(label, dtype=torch.long)
#         self.transform = transform

#     def __len__(self):
#         return 1  # Each instance is a bag containing multiple images

#     def __getitem__(self, idx):
#         images = [Image.open(p).convert('RGB') for p in self.image_paths]
#         if self.transform:
#             images = [self.transform(image) for image in images]
#         images = torch.stack(images)  # Stack images to create a single batch-like tensor
#         return images, self.label

class HistopathologyDataset(Dataset):
    def __init__(self, dataframe, transform=None, num_classes=2):
        self.dataframe = dataframe
        self.transform = transform
        self.num_classes = num_classes
        # Assume all patches in the dataframe have the same label, get the first label as the bag label
        self.bag_label = torch.tensor(dataframe['label'].iloc[0], dtype=torch.long)

    def __len__(self):
        return 1  # There is only one bag in this dataset

    def __getitem__(self, idx):
        # Load all images in the bag
        images = [Image.open(row['path']).convert("RGB") for _, row in self.dataframe.iterrows()]

        if self.transform:
            images = [self.transform(image) for image in images]

        images_tensor = torch.stack(images)  # Stack images to create a single batch-like tensor

        # One hot encode the label if necessary
        label_one_hot = self.one_hot_encode(self.bag_label, self.num_classes)

        return images_tensor, label_one_hot

    def one_hot_encode(self, label, num_classes):
        # One-hot encoding is done here, ensuring it is suitable for the number of classes
        return torch.nn.functional.one_hot(label, num_classes=num_classes)


def get_task(na=None):
    dim_y = 2
    task = mk_task_dset(isize=ImSize(3, 224, 224), dim_y=dim_y, taskna="custom_histopathology_task")
    diagnosis_to_label = {'nonMSIH': 0, 'MSIH': 1}
    img_trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    base_path = '/lustre/groups/shared/users/peng_marr/TCGA-CRC'
    metadata_path = '/lustre/groups/aih/sina.wendrich/MA_code'
    excel_path = '/lustre/groups/shared/users/peng_marr/TCGA-CRC/TCGA-CRC-DX_CLINI.xlsx'
    output_folder = '/lustre/groups/shared/users/peng_marr/TCGA_hospitals/output_CRC_patches'
    biospecimen_files = {'COAD': 'TCGA_CRC/biospecimen.project-tcga-coad.2024-05-01.json', 'READ': 'TCGA_CRC/biospecimen.project-tcga-read.2024-05-23.json'}
    center_dicts = load_center_dict(metadata_path, biospecimen_files)
    msi_status_dict = read_msi_status(excel_path)

    slide_folder = os.path.join(base_path, 'slides')
    slide_files = sorted([f for f in os.listdir(slide_folder) if f.endswith(".svs")])
    domain_datasets_train = {}
    domain_datasets_val = {}

    for i, slide_file in enumerate(slide_files):
        if i > 20:
            break
        slide_path = os.path.join(slide_folder, slide_file)
        submitter_id_parts = slide_file.split('-')
        submitter_id = '-'.join(submitter_id_parts[:3])
        MSI_status = msi_status_dict.get(submitter_id)
        print(MSI_status)
        if MSI_status is None or MSI_status == 'NA' or pd.isna(MSI_status):
            continue  # Skip processing if MSI status is NA or undefined
        diagnosis_label = diagnosis_to_label.get(MSI_status, na)
        print(diagnosis_label)
        domain_name = center_dicts.get(submitter_id, "Unknown")
        coords, image_paths_and_labels = process_slides_tissue_type(slide_path, output_folder, diagnosis_label, domain_name)
        dataset = HistopathologyDataset(image_paths_and_labels, transform=img_trans, num_classes=dim_y)

        if domain_name not in domain_datasets_train:
            domain_datasets_train[domain_name] = []
            domain_datasets_val[domain_name] = []

        if i % 4 == 0:  # Every fourth slide goes into the validation set
            domain_datasets_val[domain_name].append(dataset)
        else:
            domain_datasets_train[domain_name].append(dataset)

    for domain_name in domain_datasets_train:
        print(domain_name)
        combined_dataset_train = ConcatDataset(domain_datasets_train[domain_name])
        combined_dataset_val = ConcatDataset(domain_datasets_val[domain_name])
        task.add_domain(name=domain_name, dset_tr=combined_dataset_train, dset_val=combined_dataset_val)

    return task


def load_center_dict(metadata_path, biospecimen_files):
    center_dict = {}
    for disease, filename in biospecimen_files.items():
        biospecimen_path = os.path.join(metadata_path, filename)
        with open(biospecimen_path, 'r') as json_file:
            biospecimen_data = json.load(json_file)
        for item in biospecimen_data:
            submitter_id = item.get("submitter_id")
            if not submitter_id:
                continue
            samples = item.get("samples", [])
            if samples:
                portions = samples[0].get("portions", [])
                if portions:
                    analytes = portions[0].get("analytes", [])
                    if analytes:
                        aliquots = analytes[0].get("aliquots", [])
                        if aliquots:
                            center = aliquots[0].get("center")
                            if center:
                                center_name = center.get("name")
                                if center_name:
                                    center_dict[submitter_id] = center_name
    return center_dict

def read_msi_status(excel_path):
    df = pd.read_excel(excel_path)
    msi_status_dict = df.set_index('PATIENT')['isMSIH'].to_dict()
    return msi_status_dict