import os
import glob
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

# User parameters
base_input_folder = r"/home/crack/Road-Anomaly-Detector/road_anomaly_detector/main/miscellaneous/tiles_folder"

# Prompt the user for the target subfolder name
subfolder_name = input("Please enter the name of the target subfolder: \n").strip()

# Construct the full target directory path
input_folder = os.path.join(base_input_folder, subfolder_name)

# Create the PCSM folder structure
pcsm_folder = os.path.join("PCSM", subfolder_name)
os.makedirs(pcsm_folder, exist_ok=True)

# Create annotated images folder inside PCSM/subfolder_name
output_annotated_folder = os.path.join(pcsm_folder, 'annotated_images')
os.makedirs(output_annotated_folder, exist_ok=True)

# Set output CSV paths
output_csv = os.path.join(pcsm_folder, 'pavement_cracking_metrics.csv')
road_summary_csv = os.path.join(pcsm_folder, 'road_level_cracking_metrics.csv')

min_crack_length = 100
# One pixel equals 0.7494 mm = 0.0007494 m
pixel_spacing = 0.0007494  # meters per pixel

# Data structures to store per-road accumulations
roads_data = {}  # {road_name: {'total_crack_pixels': int, 'total_crack_length_pixels': int, 'total_pixels': int}}

# Open CSV and write header
with open(output_csv, 'w') as f:
    f.write('filename,road_name,crack_density,crack_length_density\n')

# Get all image files
image_files = glob.glob(os.path.join(input_folder, '*.png')) + \
              glob.glob(os.path.join(input_folder, '*.jpg')) + \
              glob.glob(os.path.join(input_folder, '*.jpeg'))

for img_path in image_files:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not read {img_path}, skipping.")
        continue

    filename = os.path.basename(img_path)
    # Extract road name by taking first two parts of filename split by underscore
    # e.g., "cykel_20_00000.png" -> "cykel_20"
    parts = filename.split('_')
    if len(parts) < 3:
        # If naming convention doesn't match, just use the entire filename as road_name
        road_name = filename
    else:
        road_name = '_'.join(parts[:2])

    h, w, c = img.shape
    half_w = w // 2

    # Extract original pavement image (left half) and mask (right half)
    pavement_img = img[:, :half_w]
    mask_img = img[:, half_w:]

    # Convert mask to grayscale if needed
    if mask_img.ndim == 3 and mask_img.shape[2] > 1:
        mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_img

    # Threshold the mask to ensure binary
    _, mask_bin = cv2.threshold(mask_gray, 127, 1, cv2.THRESH_BINARY)

    # Skeletonize the mask
    skeleton = skeletonize(mask_bin.astype(bool))

    # Label connected components in the skeleton
    labeled_skel = label(skeleton)

    total_crack_pixels = 0
    total_crack_length_pixels = 0
    total_pixels = mask_bin.size

    bounding_boxes = []

    for region in regionprops(labeled_skel):
        crack_length_pixels = len(region.coords)
        if crack_length_pixels >= min_crack_length:
            # Valid crack region
            # Using skeleton length as proxy for both crack density and crack length calculations
            total_crack_pixels += crack_length_pixels
            total_crack_length_pixels += crack_length_pixels

            # Bounding box in skeleton coordinates
            minr, minc, maxr, maxc = region.bbox
            bounding_boxes.append((minr, minc, maxr, maxc))

    # Compute metrics
    # Crack density (dimensionless) = total_crack_pixels / total_pixels
    if total_pixels > 0:
        crack_density = total_crack_pixels / total_pixels
    else:
        crack_density = 0.0

    # Compute crack length density in real units if pixel_spacing is known
    # crack_length (m) = total_crack_length_pixels * pixel_spacing
    # total_area (m^2) = total_pixels * (pixel_spacing^2)
    if total_pixels > 0:
        crack_length = total_crack_length_pixels * pixel_spacing  # in meters
        total_area_m2 = total_pixels * (pixel_spacing**2)
        if total_area_m2 > 0:
            crack_length_density = crack_length / total_area_m2
        else:
            crack_length_density = 0.0
    else:
        crack_length_density = 0.0

    # Print for debugging
    print(f"File: {img_path}")
    print(f"Road: {road_name}")
    print(f"Crack Density: {crack_density:.6f}")
    print(f"Crack Length Density: {crack_length_density:.6f}")

    # Write per-image metrics
    with open(output_csv, 'a') as f:
        f.write(f"{filename},{road_name},{crack_density},{crack_length_density}\n")

    # Update road accumulations
    if road_name not in roads_data:
        roads_data[road_name] = {
            'total_crack_pixels': 0,
            'total_crack_length_pixels': 0,
            'total_pixels': 0
        }

    roads_data[road_name]['total_crack_pixels'] += total_crack_pixels
    roads_data[road_name]['total_crack_length_pixels'] += total_crack_length_pixels
    roads_data[road_name]['total_pixels'] += total_pixels

    # Draw bounding boxes
    annotated_img = img.copy()
    for (minr, minc, maxr, maxc) in bounding_boxes:
        start_point = (minc + half_w, minr)
        end_point = (maxc + half_w, maxr)
        color = (0, 0, 255)
        thickness = 5
        cv2.rectangle(annotated_img, start_point, end_point, color, thickness)

    out_path = os.path.join(output_annotated_folder, filename)
    cv2.imwrite(out_path, annotated_img)

# After processing all images, compute aggregated metrics per road
with open(road_summary_csv, 'w') as f:
    f.write('road_name,crack_density,crack_length_density\n')
    for road_name, data in roads_data.items():
        total_crack_pixels = data['total_crack_pixels']
        total_crack_length_pixels = data['total_crack_length_pixels']
        total_pixels = data['total_pixels']

        if total_pixels > 0:
            agg_crack_density = total_crack_pixels / total_pixels
        else:
            agg_crack_density = 0.0

        if total_pixels > 0:
            crack_length = total_crack_length_pixels * pixel_spacing
            total_area_m2 = total_pixels * (pixel_spacing**2)
            if total_area_m2 > 0:
                agg_crack_length_density = crack_length / total_area_m2
            else:
                agg_crack_length_density = 0.0
        else:
            agg_crack_length_density = 0.0

        f.write(f"{road_name},{agg_crack_density},{agg_crack_length_density}\n")

        print(f"\nRoad: {road_name} (Aggregated)")
        print(f"Total Crack Density: {agg_crack_density:.6f}")
        print(f"Total Crack Length Density: {agg_crack_length_density:.6f}")
