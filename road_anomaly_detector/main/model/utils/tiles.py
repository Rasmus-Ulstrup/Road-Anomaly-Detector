import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
from metrics.metrics import default_transform, apply_preprocessing
# Albumentations imports
import albumentations as A
from albumentations.pytorch import ToTensorV2
# ------------------------------
# Import Configuration
# ------------------------------
from config.config import Config 

# ------------------------------
# Splitting Function
# ------------------------------

def split_image_into_tiles(image_path, tile_size=512, overlap=50, save_tiles=False, output_dir='./output/tiles/images'):
    """
    Splits a large image into smaller tiles with optional overlaps and optional saving to disk
    using OpenCV.

    Args:
        image_path (str): Path to the input image.
        tile_size (int): Size of each tile (default is 512).
        overlap (int): Number of pixels to overlap between tiles (default is 50).
        save_tiles (bool): Whether to save the tiles to disk (default is False).
        output_dir (str): Directory to save the split tiles if saving is enabled.

    Returns:
        list of dict: Each dict contains:
            - 'tile' (np.ndarray): The image tile as a NumPy array.
            - 'position' (tuple): (left, upper, right, lower) coordinates of the tile.
            - 'tile_path' (str, optional): Path where the tile was saved (if save_tiles=True).
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found at {image_path}")

    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load the image at {image_path}")

    tiles = []
    height, width = img.shape

    # Validate image size
    if width < tile_size or height < tile_size:
        raise ValueError(
            f"Image size should be at least {tile_size}x{tile_size} pixels. "
            f"Current size is {width}x{height}."
        )

    # Calculate number of tiles horizontally and vertically
    stride = tile_size - overlap
    num_tiles_x = (width - overlap) // stride
    num_tiles_y = (height - overlap) // stride

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            left = j * stride
            upper = i * stride
            right = left + tile_size
            lower = upper + tile_size

            # Handle edge cases (if the tile extends beyond image boundaries)
            if right > width:
                right = width
                left = max(width - tile_size, 0)
            if lower > height:
                lower = height
                upper = max(height - tile_size, 0)

            # Extract the tile using NumPy slicing
            tile = img[upper:lower, left:right]

            tile_info = {
                'tile': tile,
                'position': (left, upper, right, lower)
            }

            # Optionally save the tile to disk
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
    """
    Run inference on a single image tile and return the predicted mask.

    Args:
        Config: Configuration object.
        model (torch.nn.Module): The trained model.
        tile (PIL.Image): The image tile.
        device (torch.device): Device to run the inference on (CPU/GPU).

    Returns:
        numpy array: The mask array.
    """
    try:
        model.eval()
        with torch.no_grad():
            if preprocessing == True:
                tile = apply_preprocessing(tile)
                #print("preprocessing")
        
            # cv2.imwrite("./output/tile1.png",tile)
            # Define the transform
            transform = default_transform(Config)

            # Apply transformations
            transformed = transform(image=tile)
            input_tensor = transformed["image"].unsqueeze(0).to(device)
            input_tensor = input_tensor.float().to(device) / 255

            # Perform inference
            output = model(input_tensor)

            # Assuming output is a single-channel mask
            output_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)  # Binarize the output

            # Scale mask to 0-255
            mask_array = output_mask * 255

            return mask_array

    except Exception as e:
        print(f"Error during inference on a tile: {e}")
        raise

def run_inference_on_tiles(Config, model, device, tiles, save_masks=False, masks_output_dir="./output/tiles/masks", max_workers=4, preprocessing=False):
    """
    Runs inference on all tiles provided with optional saving of mask tiles.

    Args:
        Config: Configuration object.
        model (torch.nn.Module): The trained model.
        device (torch.device): Device to run the inference on (CPU/GPU).
        tiles (list of dict): List containing 'tile' and 'position' (and optionally 'tile_path').
        save_masks (bool): Whether to save the mask tiles to disk (default is False).
        masks_output_dir (str): Directory to save the mask tiles if saving is enabled.
        max_workers (int): Maximum number of parallel workers (default is 4).

    Returns:
        list of dict: Each dict contains 'mask' and 'position' (and optionally 'mask_path').
    """
    mask_results = []

    def process_tile(tile_info):
        tile = tile_info['tile']
        position = tile_info['position']
        mask = run_inference(Config, model, tile, device, preprocessing)

        mask_info = {
            'mask': mask,
            'position': position
        }

         # Optionally save the mask to disk
        if save_masks:
            os.makedirs(masks_output_dir, exist_ok=True)
            if 'tile_path' in tile_info:
                base_name = os.path.splitext(os.path.basename(tile_info['tile_path']))[0]
            else:
                # Generate a unique identifier if tile_path is not available
                base_name = f"tile_{position[0]}_{position[1]}"
            mask_path = os.path.join(masks_output_dir, f"{base_name}_mask.png")

            # Save using OpenCV instead of Pillow
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
    """
    Combines mask tiles into a single large mask image with blending in overlapping regions.

    Args:
        original_image_path (str): Path to the original large image to determine size.
        masks (list of dict): Each dict contains 'mask' (numpy array) and 'position' (tuple).
        output_image_path (str): Path to save the combined mask image.
        tile_size (int): Size of each tile (default is 512).
        overlap (int): Number of overlapping pixels between tiles (default is 50).
    """
    if not os.path.isfile(original_image_path):
        raise FileNotFoundError(f"Original image not found at {original_image_path}")

    try:
        original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            raise ValueError(f"Could not load the original image from {original_image_path}")

        orig_height, orig_width = original_image.shape  # shape is (height, width)

        # Calculate stride
        stride = tile_size - overlap

        # Initialize arrays for the combined mask and weight map
        combined_mask = np.zeros((orig_height, orig_width), dtype=np.float32)
        weight_map = np.zeros((orig_height, orig_width), dtype=np.float32)

        for mask_info in masks:
            mask_array = mask_info['mask']
            left, upper, right, lower = mask_info['position']

            # Ensure the mask fits within the original image dimensions
            right = min(right, orig_width)
            lower = min(lower, orig_height)
            left = max(right - tile_size, 0)
            upper = max(lower - tile_size, 0)

            # Define blending weights
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

            # Add to combined mask and weight map
            combined_mask[upper:lower, left:right] += mask_array * weight
            weight_map[upper:lower, left:right] += weight

        # Avoid division by zero
        weight_map[weight_map == 0] = 1.0
        final_mask = combined_mask / weight_map
        final_mask = np.clip(final_mask, 0, 255).astype(np.uint8)

        # Convert to PIL Image and save
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        cv2.imwrite(output_image_path, final_mask)
        print(f"Combined mask with blending saved to {output_image_path}")

        # Optional: Validate the combined mask by checking dimensions
        if (final_mask.shape[1], final_mask.shape[0]) != (orig_width, orig_height):
            print("Warning: Combined mask size does not match original image size.")
        else:
            print("Combined mask size matches the original image.")

        # Optional: Combine original image and mask side by side
        try:
            original_image_2 = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
            if original_image_2 is None:
                raise ValueError(f"Could not load the original image from {original_image_path}")

            # Resize original image to match mask if necessary
            if (original_image_2.shape[1], original_image_2.shape[0]) != (final_mask.shape[1], final_mask.shape[0]):
                original_image_2 = cv2.resize(
                    original_image_2,
                    (final_mask.shape[1], final_mask.shape[0]),
                    interpolation=cv2.INTER_LANCZOS4
                )

            # Combine side by side
            combined_side_by_side = np.hstack((original_image_2, final_mask))

            # Define the path for the combined image
            combined_image_path = os.path.join(
                os.path.dirname(output_image_path),
                f"combined_{os.path.basename(original_image_path)}"
            )

            # Save the combined image using OpenCV
            cv2.imwrite(combined_image_path, combined_side_by_side)
            print(f"Combined original image and mask saved to {combined_image_path}")

        except Exception as e:
            print(f"Error while combining original image and mask: {e}")
            raise

    except Exception as e:
        print(f"Error while combining tiles with blending: {e}")
        raise

# ------------------------------
# Main Processing Function
# ------------------------------

def run_main_tiles(Config, image_dir, output_dir, model, device, tile_size=512, overlap=50, max_workers=4, save_tiles=False, save_masks=False, preprocessing=False):
    """
    Processes all images in the specified directory: splits into tiles, runs inference, and combines masks.

    Args:
        Config: Configuration object.
        image_dir (str): Directory containing input images.
        output_dir (str): Directory for base outputs.
        model (torch.nn.Module): The trained model.
        device (torch.device): Device to run the inference on (CPU/GPU).
        tile_size (int): Size of each tile (default: 512).
        overlap (int): Overlap between tiles in pixels (default: 50).
        max_workers (int): Maximum number of parallel workers for inference (default: 4).
        save_tiles (bool): Whether to save image tiles to disk (default: False).
        save_masks (bool): Whether to save mask tiles to disk (default: False).
    """
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

        # Define output paths for this image
        combined_mask_dir = os.path.join(output_dir, base_name, 'combined_masks')
        combined_mask_path = os.path.join(combined_mask_dir, f"{base_name}_combined_mask.png")

        # Define tiles output directories if saving is enabled
        image_output_dir = os.path.join(output_dir, base_name, 'tiles', 'images') if save_tiles else None
        masks_output_dir = os.path.join(output_dir, base_name, 'tiles', 'masks') if save_masks else None

        try:
            # Step 1: Split image into tiles (in-memory with optional saving)
            tiles = split_image_into_tiles(
                image_path=image_path,
                tile_size=tile_size,
                overlap=overlap,
                save_tiles=save_tiles,
                output_dir=image_output_dir if save_tiles else './output/tiles/images'
            )
            print(f"Split into {len(tiles)} tiles.")

            # Step 2: Run inference on tiles (in-memory with optional saving)
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

            # Step 3: Combine masks with blending (in-memory)
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
            # Optionally, you can continue with the next image or halt execution
            continue

def run_main_tiles_metrics(Config, image_dir, mask_dir, output_dir, model, device, tile_size=512, overlap=50, max_workers=4, save_tiles=False, save_masks=False, preprocessing=False):
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

        # Define output paths for this image
        combined_mask_dir = os.path.join(output_dir, base_name, 'combined_masks')
        combined_mask_path = os.path.join(combined_mask_dir, f"{base_name}_combined_mask.png")

        # Define tiles output directories if saving is enabled
        image_output_dir = os.path.join(output_dir, base_name, 'tiles', 'images') if save_tiles else None
        masks_output_dir = os.path.join(output_dir, base_name, 'tiles', 'masks') if save_masks else None

        try:
            # Step 1: Split image into tiles (in-memory with optional saving)
            tiles = split_image_into_tiles(
                image_path=image_path,
                tile_size=tile_size,
                overlap=overlap,
                save_tiles=save_tiles,
                output_dir=image_output_dir if save_tiles else './output/tiles/images'
            )
            print(f"Split into {len(tiles)} tiles.")

            # Step 2: Run inference on tiles (in-memory with optional saving)
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

            # Step 3: Combine masks with blending (in-memory)
            combine_tiles_into_image_with_blending(
                original_image_path=image_path,
                masks=masks,
                output_image_path=combined_mask_path,
                tile_size=tile_size,
                overlap=overlap
            )

            print(f"Finished processing image: {image_path}")
            print(f"Starting metrics calculation for: {image_path}")

            print("Splitting mask into tiles")

            

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Optionally, you can continue with the next image or halt execution
            continue

# ------------------------------
# Main Execution Block
# ------------------------------

if __name__ == '__main__':
    print("Starting in-memory tile processing pipeline...")