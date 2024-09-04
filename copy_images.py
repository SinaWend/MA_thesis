import os
import shutil
import random
from PIL import Image

def copy_images_to_target(destination_folder, domain_mappings, root_paths, num_images=2):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for domain, categories in domain_mappings.items():
        root_path = root_paths[domain]
        for category, folder_name in categories.items():
            source_folder = os.path.join(root_path, folder_name)
            if os.path.exists(source_folder):
                all_images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
                selected_images = random.sample(all_images, min(num_images, len(all_images)))
                for image in selected_images:
                    source_image_path = os.path.join(source_folder, image)
                    # Open the image and convert to PNG
                    img = Image.open(source_image_path)
                    destination_image_path = os.path.join(destination_folder, f"{domain}_{category}_{os.path.splitext(image)[0]}.png")
                    img.save(destination_image_path, 'PNG')
                    print(f"Converted and copied {source_image_path} to {destination_image_path}")
            else:
                print(f"Source folder {source_folder} does not exist.")

domain_mappings = {
    "acevedo": {
        "basophil": "basophil",
        "erythroblast": "erythroblast",
        "metamyelocyte": "metamyelocyte",
        "neutrophil_band": "neutrophil_band",
        "promyelocyte": "promyelocyte",
        "eosinophil": "eosinophil",
        "lymphocyte_typical": "lymphocyte_typical",
        "monocyte": "monocyte",
        "myelocyte": "myelocyte",
        "neutrophil_segmented": "neutrophil_segmented",
    },
    "matek": {
        "basophil": "basophil",
        "erythroblast": "erythroblast",
        "metamyelocyte": "metamyelocyte",
        "myeloblast": "myeloblast",
        "neutrophil_band": "neutrophil_band",
        "promyelocyte": "promyelocyte",
        "eosinophil": "eosinophil",
        "lymphocyte_typical": "lymphocyte_typical",
        "monocyte": "monocyte",
        "myelocyte": "myelocyte",
        "neutrophil_segmented": "neutrophil_segmented",
    },
    "mll": {
        "basophil": "basophil",
        "erythroblast": "erythroblast",
        "metamyelocyte": "metamyelocyte",
        "myeloblast": "myeloblast",
        "neutrophil_band": "neutrophil_band",
        "promyelocyte": "promyelocyte",
        "eosinophil": "eosinophil",
        "lymphocyte_typical": "lymphocyte_typical",
        "monocyte": "monocyte",
        "myelocyte": "myelocyte",
        "neutrophil_segmented": "neutrophil_segmented",
    },
}

root_paths = {
    "matek": "/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Matek_cropped",
    "mll": "/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/MLL_20221220",
    "acevedo": "/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Acevedo_cropped",
}

destination_folder = "/home/aih/sina.wendrich/MA_thesis/BLOOD_samples_new"
copy_images_to_target(destination_folder, domain_mappings, root_paths)
