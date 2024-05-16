from domainlab.tasks.task_dset import mk_task_dset
import os
import json
import torch
from PIL import Image
from torchvision import transforms
from domainlab.tasks.utils_task import ImSize
from examples.tasks.patches_processing import process_slides_primary_diagnosis
from torch.utils.data import ConcatDataset, Dataset

class HistopathologyDataset(Dataset):
    def __init__(self, dataframe, transform=None, num_classes=15):
        self.dataframe = dataframe
        self.transform = transform
        self.num_classes = num_classes
        self.targets = torch.tensor(self.dataframe['label'].values, dtype=torch.long)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']
        label = self.dataframe.iloc[idx]['label']
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        label_one_hot = self.one_hot_encode(label, self.num_classes)

        return image, label_one_hot

    @staticmethod
    def one_hot_encode(labels, num_classes):
        return torch.nn.functional.one_hot(labels, num_classes=num_classes)

def get_task(na=None):
    dim_y = 15
    task = mk_task_dset(isize=ImSize(3, 224, 224), dim_y=dim_y, taskna="custom_histopathology_task")
    diagnosis_to_label = {}

    biospecimen_path = "/lustre/groups/aih/sina.wendrich/MA_code/TCGA_BRCA/biospecimen.project-tcga-brca.2024-04-30.json"
    #biospecimen_path = "data/TCGA_BRCA_metadata/biospecimen/biospecimen.project-tcga-brca.2024-04-30.json"
    with open(biospecimen_path) as json_file:
        biospecimen_data = json.load(json_file)

    center_dict = {}
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

    img_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    domain_datasets_train = {}
    domain_datasets_val = {}
    print("Constructed center_dict:", center_dict)  # Diagnostic print to check the output
    slide_folder = '/lustre/groups/shared/histology_data/TCGA/BRCA/slides'
    #slide_folder = "data/TCGA_BRCA"
    slide_files = os.listdir(slide_folder)
    for slide_file in slide_files:
        if slide_file.endswith(".svs"):
            submitter_id_parts = slide_file.split('-')
            submitter_id = '-'.join(submitter_id_parts[:3])
            if submitter_id in center_dict:
                print("Match found in center_dict for:", submitter_id)
                domain_name = center_dict[submitter_id]

                slide_path = os.path.join(slide_folder, slide_file)
                save_path = '/localscratch/sina.wendrich/output_TCGA_BRCA'

                #save_path = os.path.join(slide_folder, "output_test")
                coords, data = process_slides_primary_diagnosis(slide_path, save_path, diagnosis_to_label)

                if len(data) > 10:
                    data = data.sample(n=10).reset_index(drop=True)

                df_train, df_val, df_test = split_dataset(data, 0.7, 0.3)

                dataset_train = HistopathologyDataset(df_train, transform=img_trans, num_classes=dim_y)
                dataset_val = HistopathologyDataset(df_val, transform=img_trans, num_classes=dim_y)

                if domain_name not in domain_datasets_train:
                    domain_datasets_train[domain_name] = []
                    domain_datasets_val[domain_name] = []
                domain_datasets_train[domain_name].append(dataset_train)
                domain_datasets_val[domain_name].append(dataset_val)

    print(diagnosis_to_label)
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
