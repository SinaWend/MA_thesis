import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
import os
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

class HistopathologyDataset(Dataset):
    def __init__(self, dataframe, transform=None, num_classes=2, max_patches=None):
        self.dataframe = dataframe
        self.transform = transform
        self.num_classes = num_classes
        self.bag_label = torch.tensor(dataframe['label'].iloc[0], dtype=torch.long)
        self.max_patches = max_patches

    def __len__(self):
        return 1  # There is only one bag in this dataset

    def __getitem__(self, idx):
        images = []
        for _, row in self.dataframe.iterrows():
            try:
                img = Image.open(row['path']).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except FileNotFoundError:
                print(f"Error: File not found at {row['path']}")
                continue  # Skip the missing files
        if not images:
            raise ValueError("No valid images in the bag.")
        
        # Stack the images and pad them to ensure they are the same size
        images_tensor = torch.stack(images)
        if self.max_patches:
            images_tensor = self.pad_tensor(images_tensor, self.max_patches)
        
        label_one_hot = self.one_hot_encode(self.bag_label, self.num_classes)
        return images_tensor, label_one_hot

    def pad_tensor(self, tensor, max_patches):
        # Pad the tensor to have the same number of patches
        pad_size = max_patches - tensor.size(0)
        return torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, pad_size))

    def one_hot_encode(self, label, num_classes):
        return torch.nn.functional.one_hot(label, num_classes=num_classes)

def get_max_patches(slide_files, base_path, output_folder, diagnosis_to_label, center_dicts, msi_status_dict):
    max_patches = 0
    for slide_file in slide_files:
        slide_path = os.path.join(base_path, 'slides', slide_file)
        submitter_id_parts = slide_file.split('-')
        submitter_id = '-'.join(submitter_id_parts[:3])
        MSI_status = msi_status_dict.get(submitter_id)
        diagnosis_label = diagnosis_to_label.get(MSI_status, None)
        domain_name = center_dicts.get(submitter_id, "Unknown")
        _, image_paths_and_labels = process_slides_tissue_type(slide_path, output_folder, diagnosis_label, domain_name)
        num_patches = len(image_paths_and_labels)
        if num_patches > max_patches:
            max_patches = num_patches
    return max_patches

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

    # Limit the number of WSIs to a subset (21 in this case)
    subset_slide_files = slide_files[:21]

    max_patches = get_max_patches(subset_slide_files, base_path, output_folder, diagnosis_to_label, center_dicts, msi_status_dict)

    domain_datasets_train = {}
    domain_datasets_val = {}

    for i, slide_file in enumerate(subset_slide_files):
        slide_path = os.path.join(slide_folder, slide_file)
        submitter_id_parts = slide_file.split('-')
        submitter_id = '-'.join(submitter_id_parts[:3])
        MSI_status = msi_status_dict.get(submitter_id)
        diagnosis_label = diagnosis_to_label.get(MSI_status, na)
        if diagnosis_label is None:
            print(f"Warning: No diagnosis label found for {submitter_id}")
            continue
        domain_name = center_dicts.get(submitter_id, "Unknown")

        coords, image_paths_and_labels = process_slides_tissue_type(slide_path, output_folder, diagnosis_label, domain_name)
        if image_paths_and_labels.empty:  # Skip empty dataframes
            print(f"Warning: No valid images for {slide_file}")
            continue

        dataset = HistopathologyDataset(image_paths_and_labels, transform=img_trans, num_classes=dim_y, max_patches=max_patches)
        if domain_name not in domain_datasets_train:
            domain_datasets_train[domain_name] = []
            domain_datasets_val[domain_name] = []

        if i % 4 == 0:
            domain_datasets_val[domain_name].append(dataset)
        else:
            domain_datasets_train[domain_name].append(dataset)

    for domain_name in domain_datasets_train:
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
