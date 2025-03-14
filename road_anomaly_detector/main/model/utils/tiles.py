import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
from metrics.metrics import default_transform, apply_preprocessing, compute_metrics, compute_centerline_metrics
from sklearn.metrics import f1_score
# Albumentations imports
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config.config import Config 

def split_image_into_tiles(image_path, tile_size=512, overlap=50, save_tiles=False, output_dir='./output/tiles/images'):
    #Functuion to split images into tiles with or without overlab and save them.
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found at {image_path}")

    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load the image at {image_path}")

    tiles = []
    height, width = img.shape

    # Validate image sizes
    if width < tile_size or height < tile_size:
        raise ValueError(
            f"Image size should be at least {tile_size}x{tile_size} pixels. "
            f"Current size is {width}x{height}."
        )

    # Calculate number of tiles 
    stride = tile_size - overlap
    num_tiles_x = (width - overlap) // stride
    num_tiles_y = (height - overlap) // stride

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            left = j * stride
            upper = i * stride
            right = left + tile_size
            lower = upper + tile_size

            # Handle edge cases (beoynd image border)
            if right > width:
                right = width
                left = max(width - tile_size, 0)
            if lower > height:
                lower = height
                upper = max(height - tile_size, 0)

            # Extract the tile
            tile = img[upper:lower, left:right]

            tile_info = {
                'tile': tile,
                'position': (left, upper, right, lower)
            }

            #save tile at defined path
            if save_tiles:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                tile_filename = f"{base_name}_tile_{i}_{j}.png"
                tile_path = os.path.join(output_dir, tile_filename)
                
                # Save the tile
                cv2.imwrite(tile_path, tile)
                tile_info['tile_path'] = tile_path

            tiles.append(tile_info)

    return tiles

# ------------------------------
# Inference Functions
# ------------------------------

def run_inference(Config, model, tile, device, preprocessing):
    try:
        model.eval()
        with torch.no_grad():
            if preprocessing == True:
                tile = apply_preprocessing(tile)

            transform = default_transform(Config)

            # Apply transformations
            transformed = transform(image=tile)
            input_tensor = transformed["image"].unsqueeze(0).to(device)
            input_tensor = input_tensor.float().to(device) / 255

            #BEAR
            outputs = model(input_tensor)
            if not isinstance(outputs, list):
                output_mask = (outputs.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            else:
                final_output = outputs[-1]
                # Extract final output from list (HED and FPN)
                output_mask = (final_output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            
            # Scale mask to 0-255
            mask_array = output_mask * 255

            return mask_array

    except Exception as e:
        print(f"Error during inference on a tile: {e}")
        raise

def run_inference_on_tiles(Config, model, device, tiles, save_masks=False, masks_output_dir="./output/tiles/masks", max_workers=4, preprocessing=False):

    mask_results = []

    def process_tile(tile_info):
        tile = tile_info['tile']
        position = tile_info['position']
        mask = run_inference(Config, model, tile, device, preprocessing)

        mask_info = {
            'mask': mask,
            'position': position
        }

        if save_masks:
            os.makedirs(masks_output_dir, exist_ok=True)
            if 'tile_path' in tile_info:
                base_name = os.path.splitext(os.path.basename(tile_info['tile_path']))[0]
            else:
                base_name = f"tile_{position[0]}_{position[1]}"
            mask_path = os.path.join(masks_output_dir, f"{base_name}_mask.png")

            # Save using OpenCV
            cv2.imwrite(mask_path, mask)

            mask_info['mask_path'] = mask_path

        return mask_info

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_tile, tile_info) for tile_info in tiles]
        for future in as_completed(futures):
            try:
                mask = future.result()
                mask_results.append(mask)
            except Exception as e:
                print(f"Failed to process a tile: {e}")

    return mask_results

# ------------------------------
# Combining Function with Blending
# ------------------------------

def combine_tiles_into_image_with_blending(original_image_path, masks, output_image_path='./output/tiles/combined_mask.png', tile_size=512, overlap=50):
    if not os.path.isfile(original_image_path):
        raise FileNotFoundError(f"Original image not found at {original_image_path}")

    try:
        original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            raise ValueError(f"Could not load the original image from {original_image_path}")

        orig_height, orig_width = original_image.shape

        stride = tile_size - overlap

        # Initialize arrays
        combined_mask = np.zeros((orig_height, orig_width), dtype=np.float32)
        weight_map = np.zeros((orig_height, orig_width), dtype=np.float32)

        for mask_info in masks:
            mask_array = mask_info['mask']
            left, upper, right, lower = mask_info['position']

            # Ensure the mask fit
            right = min(right, orig_width)
            lower = min(lower, orig_height)
            left = max(right - tile_size, 0)
            upper = max(lower - tile_size, 0)

            # blending weights
            weight = np.ones_like(mask_array, dtype=np.float32)

            # Calculate overlap regions for blending
            # Horizontal blending
            if left > 0 and overlap > 0:
                weight[:, :overlap] *= np.linspace(0, 1, overlap)[None, :]
            if right < orig_width and overlap > 0:
                weight[:, -overlap:] *= np.linspace(1, 0, overlap)[None, :]

            # Vertical blending
            if upper > 0 and overlap > 0:
                weight[:overlap, :] *= np.linspace(0, 1, overlap)[:, None]
            if lower < orig_height and overlap > 0:
                weight[-overlap:, :] *= np.linspace(1, 0, overlap)[:, None]

            # Add too combined mask and weight map
            combined_mask[upper:lower, left:right] += mask_array * weight
            weight_map[upper:lower, left:right] += weight

        # set to avoid division by zero
        weight_map[weight_map == 0] = 1.0
        final_mask = combined_mask / weight_map
        final_mask = np.clip(final_mask, 0, 255).astype(np.uint8)

        # converts to PIL Image and save
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        cv2.imwrite(output_image_path, final_mask)
        print(f"Combined mask with blending saved to {output_image_path}")

        if (final_mask.shape[1], final_mask.shape[0]) != (orig_width, orig_height):
            print("Warning: Combined mask size does not match original image size.")
        else:
            print("Combined mask size matches the original image.")

        try:
            original_image_2 = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
            if original_image_2 is None:
                raise ValueError(f"Could not load the original image from {original_image_path}")

            # Resize original image to match mask
            if (original_image_2.shape[1], original_image_2.shape[0]) != (final_mask.shape[1], final_mask.shape[0]):
                original_image_2 = cv2.resize(
                    original_image_2,
                    (final_mask.shape[1], final_mask.shape[0]),
                    interpolation=cv2.INTER_LANCZOS4
                )

            # Combine image and mask
            combined_side_by_side = np.hstack((original_image_2, final_mask))

            # Define the path
            combined_image_path = os.path.join(
                os.path.dirname(output_image_path),
                f"combined_{os.path.basename(original_image_path)}"
            )

            # Save combined image
            cv2.imwrite(combined_image_path, combined_side_by_side)
            print(f"Combined original image and mask saved to {combined_image_path}")

        except Exception as e:
            print(f"Error while combining original image and mask: {e}")
            raise

    except Exception as e:
        print(f"Error while combining tiles with blending: {e}")
        raise

    return final_mask

# ------------------------------
# Main Processing Function
# ------------------------------

def run_main_tiles(Config, image_dir, output_dir, model, device, tile_size=512, overlap=50, max_workers=4, save_tiles=False, save_masks=False, preprocessing=False):
    # Supported image formats
    supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

    # List and sort image files in the input directory
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(supported_formats)
    ])

    if not image_files:
        print(f"No images found in {image_dir} with formats {supported_formats}.")
        return

    print(f"Found {len(image_files)} images to process.")

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        base_name = os.path.splitext(image_file)[0]
        print(f"\nProcessing image: {image_path}")

        # Define output paths
        combined_mask_dir = os.path.join(output_dir, base_name, 'combined_masks')
        combined_mask_path = os.path.join(combined_mask_dir, f"{base_name}_combined_mask.png")

        # Define tiles output directories
        image_output_dir = os.path.join(output_dir, base_name, 'tiles', 'images') if save_tiles else None
        masks_output_dir = os.path.join(output_dir, base_name, 'tiles', 'masks') if save_masks else None

        try:
            # Step 1: Split image into tiles
            tiles = split_image_into_tiles(
                image_path=image_path,
                tile_size=tile_size,
                overlap=overlap,
                save_tiles=save_tiles,
                output_dir=image_output_dir if save_tiles else './output/tiles/images'
            )
            print(f"Split into {len(tiles)} tiles.")

            # Step 2: Run inference on tiles
            masks = run_inference_on_tiles(
                Config=Config,
                model=model,
                device=device,
                tiles=tiles,
                save_masks=save_masks,
                masks_output_dir=masks_output_dir if save_masks else "./output/tiles/masks",
                max_workers=max_workers,
                preprocessing=preprocessing
            )
            print(f"Generated {len(masks)} masks.")

            if len(masks) == 0:
                print("No masks were generated. Skipping combination step.")
                continue

            # Step 3: Combine masks with blending
            combine_tiles_into_image_with_blending(
                original_image_path=image_path,
                masks=masks,
                output_image_path=combined_mask_path,
                tile_size=tile_size,
                overlap=overlap
            )

            print(f"Finished processing image: {image_path}")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

def sort_key(d):
    return (d['position'][1], d['position'][0])

def validate_sorted_lists(sorted_list1, sorted_list2):
    if len(sorted_list1) != len(sorted_list2):
        raise ValueError("The two lists have different lengths.")
    
    for idx, (dict1, dict2) in enumerate(zip(sorted_list1, sorted_list2)):
        if dict1['position'] != dict2['position']:
            raise ValueError(
                f"Position mismatch at index {idx}:\n"
                f"Dict1 Position: {dict1['position']}\n"
                f"Dict2 Position: {dict2['position']}"
            )
    
    print("Validation Passed: Both lists are sorted and positions match correctly.")

def run_main_tiles_metrics(Config, folder_dir, output_dir, model, device, tile_size=512, overlap=50, max_workers=4, save_tiles=False, save_masks=False, preprocessing=False):
    # image formats defined
    supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

    image_dir = os.path.join(folder_dir, "images")
    mask_dir = os.path.join(folder_dir, "mask")

    # List and sort image files in the input directory
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(supported_formats)
    ])

    if not image_files:
        print(f"No images found in {image_dir} with formats {supported_formats}.")
        return
    
    mask_files = sorted([
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(supported_formats)
    ])

    if not image_files:
        print(f"No images found in {mask_dir} with formats {supported_formats}.")
        return

    print(f"Found {len(image_files)} images and {len(mask_files)} mask to process.")

    if len(image_files) != len(mask_files):
        print("Mask and images are not the same length...")
        return
    
    image_metrics = {
            "correctness": [],
            "completeness": [],
            "quality": [],
            "precision": [],
            "recall": [],
            "IoU": [],
            "f1": []
        }

    for image_file, mask_file in zip(image_files, mask_files):
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)

        base_name = os.path.splitext(image_file)[0]
        print(f"\nProcessing image: {image_path}")

        # Define output paths for image
        combined_mask_dir = os.path.join(output_dir, base_name, 'combined_masks')
        combined_mask_path = os.path.join(combined_mask_dir, f"{base_name}_combined_mask.png")

        # Define tiles output directories
        image_output_dir = os.path.join(output_dir, base_name, 'tiles', 'images') if save_tiles else None
        masks_output_dir = os.path.join(output_dir, base_name, 'tiles', 'masks') if save_masks else None

        #Metrics
        tile_correctness = []
        tile_completeness = []
        tile_quality = []
        tile_f1 = []

        try:
            # Step 1: Split image into tiles
            tiles = split_image_into_tiles(
                image_path=image_path,
                tile_size=tile_size,
                overlap=overlap,
                save_tiles=save_tiles,
                output_dir=image_output_dir if save_tiles else './output/tiles/images'
            )
            print(f"Split into {len(tiles)} tiles.")

            # Step 2: Run inference on tiles
            masks = run_inference_on_tiles(
                Config=Config,
                model=model,
                device=device,
                tiles=tiles,
                save_masks=save_masks,
                masks_output_dir=masks_output_dir if save_masks else "./output/tiles/masks",
                max_workers=max_workers,
                preprocessing=preprocessing
            )
            print(f"Generated {len(masks)} masks.")

            if len(masks) == 0:
                print("No masks were generated. Skipping combing step.")
                continue

            # Step 3: Combine masks with blending
            final_mask_prediction = combine_tiles_into_image_with_blending(
                original_image_path=image_path,
                masks=masks,
                output_image_path=combined_mask_path,
                tile_size=tile_size,
                overlap=overlap
            )

            print(f"Starting metrics calculation for: {image_path}")

            # print(f"Splitting mask into tiles for: {mask_path}")
            # mask_tiles = split_image_into_tiles(
            #     image_path=mask_path,
            #     tile_size=tile_size,
            #     overlap=overlap,
            #     save_tiles=False,
            #     output_dir=image_output_dir if save_tiles else './output/tiles/images'
            # )
            # print(f"Split mask into {len(mask_tiles)} tiles.")

            # Compute metrics
            #print("Computing metrics")
            # sorted_masks = sorted(masks, key=sort_key)
            # sorted_mask_tiles = sorted(mask_tiles, key=sort_key)

            # validate_sorted_lists(sorted_masks, sorted_mask_tiles)

            real_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if real_mask is None:
                raise ValueError(f"Could not load the image at {mask_path}")

            
            gt_binary = (real_mask > 0).astype(np.uint8)  # Non-zero values become 1
            pred_binary = (final_mask_prediction > 0).astype(np.uint8)  # Non-zero values become 1
            f1 = f1_score(gt_binary.flatten(), pred_binary.flatten(), zero_division=1)
            if f1 == 0:
                f1 = 1
            precision, recall, IoU = compute_metrics(pred_binary, gt_binary)
            correctness, completeness, quality = compute_centerline_metrics(pred_binary, gt_binary, 2)

            # for pred_dict, gt_binary_dict in zip(sorted_masks, sorted_mask_tiles):
            #     # Check postions
            #     if pred_dict['position'] != gt_binary_dict['position']:
            #         print("WARNING: position mismatch!")

            #     #  # Display the image and mask
            #     # plt.figure(figsize=(12, 6))
            #     # plt.subplot(1, 2, 1)
            #     # plt.imshow(pred_dict['mask'], cmap="gray")
            #     # plt.title("Transformed Image")
            #     # plt.axis("off")

            #     # plt.subplot(1, 2, 2)
            #     # plt.imshow(gt_binary_dict['tile'], cmap="gray")
            #     # plt.title("not transformed Image")
            #     # plt.axis("off")

            #     # print("Press any key to close the plot and continue...")
            #     # plt.show(block=True)

            #     pred = pred_dict['mask'] / 255
            #     gt_binary = gt_binary_dict['tile'] / 255
            #     #print("Prediction unquie = ", np.unique(pred))
            #     #print("mask unquie = ", np.unique(gt_binary))
            #     correctness, completeness, quality = compute_metrics(pred, gt_binary)
            #     tile_correctness.append(correctness)
            #     tile_completeness.append(completeness)
            #     tile_quality.append(quality)
            #     tile_f1.append(f1_score(gt_binary.flatten(), pred.flatten(), zero_division=1))

            # calculate average for image
            image_metrics["correctness"].append(correctness)
            image_metrics["completeness"].append(completeness)
            image_metrics["quality"].append(quality)
            image_metrics["precision"].append(precision)
            image_metrics["recall"].append(recall)
            image_metrics["IoU"].append(IoU)
            image_metrics["f1"].append(f1)

            # print(f"Calculated metrics for image {image_path}")
            # print(f"Correct {correctness}")
            # print(f"Complete {completeness}")
            # print(f"Qual {quality}")
            # print(f"F {f1}")
            
             
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

    # Print matrics
    print("---------------------------------")
    print("              METRICS            ")
    print("---------------------------------")
    for metric, values in image_metrics.items():
        avg_value = np.mean(values)
        print(f"{metric.capitalize()}: {avg_value:.4f}")

    return image_metrics

# ------------------------------
# Main Execution Block
# ------------------------------

if __name__ == '__main__':
    print("Starting in-memory tile processing pipeline...")