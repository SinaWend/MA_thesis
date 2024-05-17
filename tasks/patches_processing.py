import json
import concurrent.futures
import time
from pathlib import Path
import numpy as np
import pandas as pd
import slideio
import torch
from PIL import Image
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, box
import matplotlib.pyplot as plt
import os



import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import xml.etree.ElementTree as ET



def get_driver(extension_name: str):
    """
    Determine the driver to use for opening an image file based on its extension.

    Args:
    - extension_name: a string representing the file extension of the image file.

    Returns:
    - A string representing the driver to use for opening the image file.
    """

    if extension_name in [".tiff", ".tif", ".jpg", ".jpeg", ".png"]:
        return "GDAL"
    elif extension_name == "":
        return "DCM"
    else:
        return extension_name.replace(".", "").upper()
    

def get_scaling(downscaling_factor, resolution_in_mpp, mpp_resolution_slide: float):
    """
    Determine the scaling factor to apply to an image based on the desired resolution in micrometers per pixel and the
    resolution in micrometers per pixel of the slide.

    Args:
    - args: a namespace containing the following attributes:
        - downscaling_factor: a float representing the downscaling factor to apply to the image.
        - resolution_in_mpp: a float representing the desired resolution in micrometers per pixel.
    - mpp_resolution_slide: a float representing the resolution in micrometers per pixel of the slide.

    Returns:
    - A float representing the scaling factor to apply to the image.
    """

    if downscaling_factor > 0:
        return downscaling_factor
    else:
        return resolution_in_mpp / (mpp_resolution_slide * 1e06)
    

def threshold(patch: np.array, white_thresh, black_thresh, calc_thresh, invalid_ratio_thresh, edge_threshold):
    """
    Determine if a patch of an image should be considered invalid based on the following criteria:
    - The number of pixels with color values above a white threshold and below a black threshold should not exceed
    a certain ratio of the total pixels in the patch.
    - The patch should have significant edges.
    If these conditions are not met, the patch is considered invalid and False is returned.

    Args:
    - patch: a numpy array representing the patch of an image.
    - args: a namespace containing at least the following attributes:
        - white_thresh: a float representing the white threshold value.
        - black_thresh: a float representing the black threshold value.
        - invalid_ratio_thresh: a float representing the maximum ratio of foreground pixels to total pixels in the patch.
        - edge_threshold: a float representing the minimum edge value for a patch to be considered valid.

    Returns:
    - A boolean value indicating whether the patch is valid or not.
    """


    

    # Count the number of whiteish pixels in the patch
    whiteish_pixels = np.count_nonzero(
        (patch[:, :, 0] > white_thresh[0])
        & (patch[:, :, 1] > white_thresh[1])
        & (patch[:, :, 2] > white_thresh[2])
    )

    # Count the number of black pixels in the patch
    black_pixels = np.count_nonzero(
        (patch[:, :, 0] <= black_thresh)
        & (patch[:, :, 1] <= black_thresh)
        & (patch[:, :, 2] <= black_thresh)
    )
    dark_pixels = np.count_nonzero(
        (patch[:, :, 0] <= calc_thresh[0])
        & (patch[:, :, 1] <= calc_thresh[1])
        & (patch[:, :, 2] <= calc_thresh[2])
    )
    calc_pixels = dark_pixels - black_pixels

    if (
        calc_pixels / (patch.shape[0] * patch.shape[1]) >= 0.05
    ):  # we always want to keep calc in!
        return True

    # Compute the ratio of foreground pixels to total pixels in the patch
    #invalid_ratio = (whiteish_pixels + black_pixels) / (patch.shape[0] * patch.shape[1])
    whiteish_pixels_float = float(whiteish_pixels)
    black_pixels_float = float(black_pixels)
    total_pixels_float = float(patch.shape[0] * patch.shape[1])

    # Compute the ratio of foreground pixels to total pixels in the patch
    invalid_ratio = (whiteish_pixels_float + black_pixels_float) / total_pixels_float
    invalid_ratio = round(invalid_ratio, 4)  # Round to 4 decimal places

    # Check if the ratio exceeds the threshold for invalid patches
    if invalid_ratio <= invalid_ratio_thresh:
        # Compute the edge map of the patch using Canny edge detection
        edge = cv2.Canny(patch, 40, 100)

        # If the maximum edge value is greater than 0, compute the mean edge value as a percentage of the maximum value
        if np.max(edge) > 0:
            edge = np.mean(edge) * 100 / np.max(edge)
        else:
            edge = 0

        # Check if the edge value is below the threshold for invalid patches or is NaN
        if (edge < edge_threshold) or np.isnan(edge):
            return False
        else:
            return True

    else:
        return False



def process_row(
    wsi: np.array, scn: int, x: int, patch_size,save_path ,downscaling_factor, slide_name: str, white_thresh, black_thresh, calc_thresh, invalid_ratio_thresh, edge_threshold
):
    """
    Process a row of a whole slide image (WSI) and extract patches that meet the threshold criteria.

    Parameters:
    wsi (numpy.ndarray): The whole slide image as a 3D numpy array (height, width, color channels).
    scn (int): Scene number of the WSI.
    x (int): X coordinate of the patch in the WSI.
    args (argparse.Namespace): Parsed command-line arguments.
    slide_name (str): Name of the slide.

    Returns:
    pd.DataFrame: A DataFrame with the coordinates of the patches that meet the threshold.
    """

    patches_coords = pd.DataFrame()
    im_paths = []
    for y in range(0, wsi.shape[1], patch_size):
        # check if a full patch still 'fits' in y direction
        if y + patch_size > wsi.shape[1]:
            continue

        # extract patch
        patch = wsi[x : x + patch_size, y : y + patch_size, :]

        # threshold checks if it meets canny edge detection, white and black pixel criteria
        if threshold(patch, white_thresh, black_thresh, calc_thresh, invalid_ratio_thresh, edge_threshold):

            im = Image.fromarray(patch)
            im_path = Path(save_path) / "patches" /str(downscaling_factor) / slide_name / f"{slide_name}_patch_{scn}_{x}_{y}.png"
            im.save(im_path)
            im_paths.append(im_path)

            patches_coords = pd.concat(
                [patches_coords, pd.DataFrame({"scn": [scn], "x": [x], "y": [y]})],
                ignore_index=True,
            )
   
    return patches_coords, im_paths 





def process_slides_tissue_type(slide_path, save_path, tissue_type, file_extension=".svs", 
                                     patch_size=128, downscaling_factor=8, resolution_in_mpp=0, 
                                     white_thresh=[175, 190, 178], black_thresh=0, calc_thresh=[40, 40, 40], 
                                     invalid_ratio_thresh=0.5, edge_threshold=4):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Slide files
    slide_files = sorted(Path(slide_path).glob(f"**/*{file_extension}"))
    print(f"Found {len(slide_files)} slide files with extension {file_extension}.")

    # Initialize data structures
    coords = pd.DataFrame({"scn": [], "x": [], "y": []}, dtype=int)
    paths = pd.DataFrame({"path": []})

    # Assign label based on passed tissue type
    label = tissue_type

    # Process each slide file
    start = time.perf_counter()
    
    driver = get_driver(file_extension)
    slide = slideio.open_slide(slide_path, driver)
    slide_name = Path(slide_path).stem
    # Process slide using your existing logic

    (Path(save_path) / "patches" / str(downscaling_factor) / slide_name).mkdir(parents=True, exist_ok=True)

    orig_sizes = []
    # iterate over scenes of the slides
    for scn in range(slide.num_scenes):
        scene = slide.get_scene(scn)
        orig_sizes.append(scene.size)
        try:
            scaling = get_scaling(downscaling_factor, resolution_in_mpp, scene.resolution[0])
        except Exception as e:
            print(e)
            print(f"Error determining resolution at slide ", slide_name, scn)
            break
        # read the scene in the desired resolution
        wsi = scene.read_block(size=(int(scene.size[0] // scaling), int(scene.size[1] // scaling)))

        # Define the main loop that processes all patches
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for x in tqdm(range(0, wsi.shape[0], patch_size), position=1, leave=False, desc=slide_name + "_" + str(scn)):
                # check if a full patch still 'fits' in x direction
                if x + patch_size > wsi.shape[0]:
                    continue
                future = executor.submit(process_row, wsi, scn, x, patch_size, save_path, downscaling_factor, slide_name, white_thresh, black_thresh, calc_thresh, invalid_ratio_thresh, edge_threshold)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                patches_coords, patches_paths = future.result()
                if len(patches_coords) > 0:
                    coords = pd.concat([coords, patches_coords], ignore_index=True)
                    paths = pd.concat([paths, pd.DataFrame({"path": patches_paths})], ignore_index=True)

    # Append data to paths DataFrame
    length = len(paths)
    paths['label'] = [label] * length  # Assign the same label to all patches from the slide

    end = time.perf_counter()
    elapsed_time = end - start
    print("Time taken: ", elapsed_time, "seconds")
    return coords, paths
  







def extract_submitter_id(full_slide_name):
    # Assumes that the submitter_id is always the first three parts of the file name
    parts = full_slide_name.split('-')
    if len(parts) >= 3:
        submitter_id = '-'.join(parts[:3])
    else:
        submitter_id = full_slide_name  # Fallback if the name is unexpectedly formatted
    return submitter_id

def process_slides_primary_diagnosis(slide_path, save_path, diagnosis_to_label, file_extension=".svs", 
                                     patch_size=128, downscaling_factor=8, resolution_in_mpp=0, 
                                     white_thresh=[175, 190, 178], black_thresh=0, calc_thresh=[40, 40, 40], 
                                     invalid_ratio_thresh=0.5, edge_threshold=4):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load clinical data
    clinical_path = "/lustre/groups/aih/sina.wendrich/MA_code/TCGA_BRCA/clinical.project-tcga-brca.2024-04-30.json"

    #clinical_path = "data/TCGA_BRCA_metadata/clinical/clinical.project-tcga-brca.2024-04-30.json"
    with open(clinical_path, 'r') as file:
        clinical_data = json.load(file)
    clinical_dict = {entry["submitter_id"]: entry for entry in clinical_data}

    # Process each slide file
    start = time.perf_counter()
    coords = pd.DataFrame({"scn": [], "x": [], "y": []}, dtype=int)
    paths = pd.DataFrame({"path": []})

    driver = get_driver(file_extension)
    slide = slideio.open_slide(slide_path, driver)
    slide_name = Path(slide_path).stem

    submitter_id = extract_submitter_id(slide_name)
    clinical_entry = clinical_dict.get(submitter_id)
    if clinical_entry and 'diagnoses' in clinical_entry and clinical_entry['diagnoses']:
        primary_diagnosis = clinical_entry['diagnoses'][0]['primary_diagnosis']
        if primary_diagnosis not in diagnosis_to_label:
            diagnosis_to_label[primary_diagnosis] = len(diagnosis_to_label)
        label = diagnosis_to_label[primary_diagnosis]
    else:
        print(f"Warning: Clinical data for {slide_name} not found.")
        label = None  # Default label indicating missing data


    (Path(save_path) / "patches" / str(downscaling_factor) / slide_name).mkdir(parents=True, exist_ok=True)

    orig_sizes = []
    # iterate over scenes of the slides
    for scn in range(slide.num_scenes):
        scene = slide.get_scene(scn)
        orig_sizes.append(scene.size)
        try:
            scaling = get_scaling(downscaling_factor, resolution_in_mpp, scene.resolution[0])
        except Exception as e:
            print(e)
            print(f"Error determining resolution at slide ", slide_name, scn)
            break
        # read the scene in the desired resolution
        wsi = scene.read_block(size=(int(scene.size[0] // scaling), int(scene.size[1] // scaling)))

        # Define the main loop that processes all patches
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for x in tqdm(range(0, wsi.shape[0], patch_size), position=1, leave=False, desc=slide_name + "_" + str(scn)):
                # check if a full patch still 'fits' in x direction
                if x + patch_size > wsi.shape[0]:
                    continue
                future = executor.submit(process_row, wsi, scn, x, patch_size, save_path, downscaling_factor, slide_name, white_thresh, black_thresh, calc_thresh, invalid_ratio_thresh, edge_threshold)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                patches_coords, patches_paths = future.result()
                if len(patches_coords) > 0:
                    coords = pd.concat([coords, patches_coords], ignore_index=True)
                    paths = pd.concat([paths, pd.DataFrame({"path": patches_paths})], ignore_index=True)

    # Append data to paths DataFrame
    length = len(paths)
    paths['label'] = [label] * length  # Assign the same label to all patches from the slide

    end = time.perf_counter()
    elapsed_time = end - start
    print("Time taken: ", elapsed_time, "seconds")
    return coords, paths




#for CAMLEYON17 task:

def parse_annotations(annotation_path,downscaling_factor):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    annotations = []
    annotation_polygons = []
    for annotation in root.findall('.//Annotations/Annotation'):
        label = annotation.attrib.get('PartOfGroup', 'unlabeled')  # Get label from XML
        points = []
        for coordinate in annotation.find('.//Coordinates'):
            x = float(coordinate.get('X')) // downscaling_factor  # Scale down the X coordinate
            y = float(coordinate.get('Y')) // downscaling_factor  # Scale down the Y coordinate
            points.append((x, y))
        polygon = MplPolygon(points, closed=True, edgecolor='g', fill=None)  # Red for visibility
        annotation_polygons.append(polygon)
        annotations.append((Polygon(points), label))
        return annotations, annotation_polygons 



def process_slides(slide_path, annotation_path, save_path,center_path, file_extension=".tif", patch_size=128, downscaling_factor=8,
                   resolution_in_mpp=0, white_thresh=[195, 210, 200], black_thresh=0, calc_thresh=[40, 40, 40],
                   invalid_ratio_thresh=0.5, edge_threshold=4):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   
    center_name = os.path.basename(center_path)

    
    driver = get_driver(file_extension)
    slide = slideio.Slide(str(slide_path), driver)
   
    annotations , annotation_polygons = parse_annotations(annotation_path, downscaling_factor)
    
    coords = pd.DataFrame({"scn": [], "x": [], "y": []}, dtype=int)
    paths = pd.DataFrame({"path": [], "label": []})

    start = time.perf_counter()
    orig_sizes = []
    for scn in range(slide.num_scenes):
        scene_coords = pd.DataFrame({"scn": [], "x": [], "y": []}, dtype=int)
        scene_paths = []
        scene = slide.get_scene(scn)
        orig_sizes.append(scene.size)
        scaling = get_scaling(downscaling_factor, resolution_in_mpp, scene.resolution[0])
        wsi = scene.read_block(
                size=(int(scene.size[0] // scaling), int(scene.size[1] // scaling))
            )        
    
#   #plot wsi:            
#         fig, ax = plt.subplots(figsize=(10, 10))
#         ax.imshow(wsi, aspect='equal')  # Display the downsampled WSI
#         ax.set_title('Downsampled WSI with Annotations')
#     # Add patches to the plot
#         p = PatchCollection(annotation_polygons, facecolor='green', edgecolor='green', alpha=0.9)
#         ax.add_collection(p)
# # Save the figure
#         plt.savefig(save_path)
#         plt.close()


        print(f"Processing scene {scn}, scaled size: {wsi.shape}")
        #The coordinate system of the wsi consists of heigth and width, so the wsi has to be indexed
        # at wsi[1] for x and wsi[0] for y
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for x in tqdm(range(0, wsi.shape[1], patch_size), position=1, leave=False, desc=f"{Path(slide_path).stem}_{scn}"):
                if x + patch_size > wsi.shape[1]:
                    continue
                future = executor.submit(process_row_annotations, wsi, scn, x, patch_size, save_path,center_name, downscaling_factor, Path(slide_path).stem, white_thresh, black_thresh, calc_thresh, invalid_ratio_thresh, edge_threshold, annotations)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                patches_coords, patches_paths = future.result()
                if patches_coords.shape[1] > 0:
                    scene_coords = pd.concat([scene_coords, patches_coords], ignore_index=True)
                    scene_paths.extend(patches_paths)

        coords = pd.concat([coords, scene_coords], ignore_index=True)
        paths = pd.concat([paths, pd.DataFrame(scene_paths)], ignore_index=True)

    end = time.perf_counter()
    print("patch count:", len(paths))
    print("Time taken: ", end - start, "seconds")
    

    labeled_patches = paths[paths['label'] == 1.0]
# Print out the number of labeled patches
    print(f"Number of labeled patches: {labeled_patches.shape[0]}")
    return coords, paths

def process_row_annotations(wsi, scn, x, patch_size, save_path, center_name, downscaling_factor, slide_name, white_thresh, black_thresh, calc_thresh, invalid_ratio_thresh, edge_threshold, annotations):
    patches_coords = pd.DataFrame({"scn": [], "x": [], "y": [], "label": []})
    im_paths = []
    for y in range(0, wsi.shape[0], patch_size):
        if y + patch_size > wsi.shape[0]:
            continue
        patch = wsi[y:y+patch_size, x:x+patch_size, :]
        if threshold(patch, white_thresh, black_thresh, calc_thresh, invalid_ratio_thresh, edge_threshold):
            patch_dir = Path(save_path) / "patches" / f"{downscaling_factor}" / slide_name 
            patch_dir.mkdir(parents=True, exist_ok=True)        
            patch_polygon = Polygon([(x , y), 
                                     ((x + patch_size) , y ),
                                     ((x + patch_size) , (y + patch_size) ),
                                     (x , (y + patch_size) )])
            label = 0
            for annotation, anns in annotations:
                if annotation.intersects(patch_polygon) or annotation.touches(patch_polygon):
                    label = 1
                    break
            im_path = patch_dir / f"{slide_name}_patch_{scn}_{x}_{y}_{center_name}_{label}.png"
            im = Image.fromarray(patch)
            im.save(im_path)

            im_paths.append({"path": str(im_path), "label": label})
            patches_coords = pd.concat([patches_coords, pd.DataFrame({"scn": [scn], "x": [x], "y": [y], "label": [label]})], ignore_index=True)

    return patches_coords, im_paths
