import os
import glob
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import re
import argparse
import logging
import math  # for exponential function

import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to extract road name using regex
def extract_road_name(filename):
    """
    Adjust this if needed. For now, we'll keep it simple: 
    Try to match pattern like 'RoadName_123.png'; if not, fallback to no extension.
    """
    match = re.match(r'^([a-zA-Z]+(?:_[a-zA-Z0-9]+)*)_\d+\.(?:png|jpg|jpeg)$', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return os.path.splitext(filename)[0]

def main():
    parser = argparse.ArgumentParser(description='Compute Pavement Surface Cracking Metrics (PSCM) and Index (PSCI)')
    parser.add_argument('--base_input_folder', type=str, required=True, help='Base input folder path')
    parser.add_argument('--subfolder_name', type=str, default="", required=False, help='Target subfolder name')
    parser.add_argument('--pixel_spacing', type=float, default=0.0007494, help='Meters per pixel')
    parser.add_argument('--min_crack_length_m', type=float, default=0.1, help='Minimum crack length in meters')
    parser.add_argument('--min_crack_width_m', type=float, default=0.001, help='Minimum crack width in meters')
    args = parser.parse_args()

    base_input_folder = args.base_input_folder
    subfolder_name = args.subfolder_name
    pixel_spacing = args.pixel_spacing
    min_crack_length_m = args.min_crack_length_m
    min_crack_width_m = args.min_crack_width_m

    logging.info(f"Base Input Folder: {base_input_folder}")
    logging.info(f"Subfolder Name: {subfolder_name}")
    logging.info(f"Pixel Spacing: {pixel_spacing} meters/pixel")
    logging.info(f"Minimum Crack Length: {min_crack_length_m} meters")
    logging.info(f"Minimum Crack Width: {min_crack_width_m} meters")

    # Calculate minimum criteria in pixels
    min_crack_length_pixels = int(np.ceil(min_crack_length_m / pixel_spacing))
    min_crack_width_pixels = int(np.ceil(min_crack_width_m / pixel_spacing))

    logging.info(f"Minimum Crack Length in Pixels: {min_crack_length_pixels}")
    logging.info(f"Minimum Crack Width in Pixels: {min_crack_width_pixels}")

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

    # Data structures to store per-road accumulations
    roads_data = {}

    # Open CSV and write header (including PSCI)
    with open(output_csv, 'w') as f:
        f.write('filename,road_name,PSCM,PSCI\n')

    # Get all image files
    image_files = (
        glob.glob(os.path.join(input_folder, '*.png')) +
        glob.glob(os.path.join(input_folder, '*.jpg')) +
        glob.glob(os.path.join(input_folder, '*.jpeg'))
    )

    logging.info(f"Found {len(image_files)} images to process.")

    for img_path in image_files:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            logging.warning(f"Could not read {img_path}, skipping.")
            continue

        filename = os.path.basename(img_path)
        road_name = extract_road_name(filename)

        logging.debug(f"Processing file: {filename}, Road: {road_name}")

        h, w, c = img.shape
        half_w = w // 2

        # Extract left half (original pavement) and right half (mask)
        pavement_img = img[:, :half_w]
        mask_img = img[:, half_w:]

        # Convert mask to grayscale if needed
        if mask_img.ndim == 3 and mask_img.shape[2] > 1:
            mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask_img

        # threshold
        _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        mask_bin = (mask_bin // 255).astype(np.uint8)

        # Distance transform for width estimation
        distance = distance_transform_edt(mask_bin)

        # Skeletonize the mask
        skeleton = skeletonize(mask_bin.astype(bool))
        # # display results
        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)

        # ax = axes.ravel()

        # ax[0].imshow(distance, cmap=plt.cm.gray)
        # ax[0].axis('off')
        # ax[0].set_title('original', fontsize=20)

        # ax[1].imshow(skeleton, cmap=plt.cm.gray)
        # ax[1].axis('off')
        # ax[1].set_title('skeleton', fontsize=20)

        # fig.tight_layout()
        # plt.show()

        # Label connected components in the skeleton
        labeled_skel = label(skeleton,connectivity=2)

        total_crack_area = 0.0  # Sum of (length_i * width_i) in m^2
        total_pixels = mask_bin.size

        bounding_boxes = []

        for region in regionprops(labeled_skel):
            crack_length_pixels = len(region.coords)
            if crack_length_pixels >= min_crack_length_pixels:
                # Mean distance * 2 is an estimate of average width
                crack_width_pixels = distance[tuple(region.coords.T)].mean() * 2

                # Convert pixels to meters
                crack_length_m = crack_length_pixels * pixel_spacing
                crack_width_m = crack_width_pixels * pixel_spacing

                # Check if crack width meets the minimum requirement
                if crack_width_m >= min_crack_width_m:
                    total_crack_area += crack_length_m * crack_width_m

                    # Keep bounding box for annotation
                    minr, minc, maxr, maxc = region.bbox
                    bounding_boxes.append((minr, minc, maxr, maxc))
                else:
                    logging.debug(
                        f"Excluded crack in {filename} due to insufficient width: {crack_width_m:.4f} m"
                    )

        # Compute PSCM
        # PSCM = 100 * (sum of crack areas) / total_area
        A = total_pixels * (pixel_spacing ** 2)  # Total area in m^2
        PSCM = 100 * total_crack_area / A if A > 0 else 0.0

        # Compute PSCI
        PSCI = 100 * math.exp(-0.45 * PSCM) if PSCM >= 0 else 0.0

        logging.info(f"File: {filename}, Road: {road_name}, PSCM: {PSCM:.6f}%, PSCI: {PSCI:.6f}%")

        # Write per-image metrics
        with open(output_csv, 'a') as f:
            f.write(f"{filename},{road_name},{PSCM:.6f},{PSCI:.6f}\n")

        # Update road accumulations
        if road_name not in roads_data:
            roads_data[road_name] = {
                'total_crack_area': 0.0,
                'total_area': 0.0
            }
        roads_data[road_name]['total_crack_area'] += total_crack_area
        roads_data[road_name]['total_area'] += A

        # Draw bounding boxes
        annotated_img = img.copy()
        for (minr, minc, maxr, maxc) in bounding_boxes:
            start_point = (minc + half_w, minr)
            end_point = (maxc + half_w, maxr)
            color = (0, 0, 255)  # Red color in BGR
            thickness = 2
            cv2.rectangle(annotated_img, start_point, end_point, color, thickness)

        out_path = os.path.join(output_annotated_folder, filename)
        print(out_path)
        cv2.imwrite(out_path, annotated_img)

    # After processing all images, compute aggregated metrics per road
    with open(road_summary_csv, 'w') as f:
        f.write('road_name,PSCM,PSCI\n')
        for road_name, data in roads_data.items():
            total_crack_area = data['total_crack_area']
            total_area = data['total_area']

            PSCM_road = 100 * total_crack_area / total_area if total_area > 0 else 0.0
            PSCI_road = 100 * math.exp(-0.45 * PSCM_road) if PSCM_road >= 0 else 0.0

            f.write(f"{road_name},{PSCM_road:.6f},{PSCI_road:.6f}\n")
            logging.info(
                f"Road: {road_name} (Aggregated), PSCM: {PSCM_road:.6f}%, PSCI: {PSCI_road:.6f}%"
            )

if __name__ == "__main__":
    main()
