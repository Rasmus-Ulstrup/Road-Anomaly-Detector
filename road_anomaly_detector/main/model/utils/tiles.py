import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------
# Import Configuration
# ------------------------------
from config.config import Config  # Ensure this path is correct based on your project structure

# ------------------------------
# Splitting Function
# ------------------------------

def split_image_into_tiles(image_path, tile_size=512, overlap=50, output_dir='./output/tiles/images'):
    """
    Splits a large image into smaller tiles with optional overlaps.

    Args:
        image_path (str): Path to the input image.
        tile_size (int): Size of each tile (default is 512).
        overlap (int): Number of pixels to overlap between tiles (default is 50).
        output_dir (str): Directory to save the split tiles.

    Returns:
        list: List of file paths to the saved tiles.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found at {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        with Image.open(image_path) as img:
            img = img.convert("L")  # Convert to grayscale if needed
            width, height = img.size

            # Validate image size
            if width < tile_size or height < tile_size:
                raise ValueError(f"Image size should be at least {tile_size}x{tile_size} pixels.")

            tiles = []
            # Calculate number of tiles horizontally and vertically
            stride = tile_size - overlap
            num_tiles_x = (width - overlap) // stride
            num_tiles_y = (height - overlap) // stride

            base_name = os.path.splitext(os.path.basename(image_path))[0]

            for i in range(num_tiles_y):
                for j in range(num_tiles_x):
                    left = j * stride
                    upper = i * stride
                    right = left + tile_size
                    lower = upper + tile_size

                    # Handle edge cases
                    if right > width:
                        right = width
                        left = max(width - tile_size, 0)
                    if lower > height:
                        lower = height
                        upper = max(height - tile_size, 0)

                    tile = img.crop((left, upper, right, lower))
                    tile_path = os.path.join(output_dir, f"{base_name}_tile_{i}_{j}.png")
                    tile.save(tile_path)
                    tiles.append(tile_path)
                    #print(f"Saved tile ({i}, {j}) to {tile_path}")

            return tiles

    except Exception as e:
        print(f"Error while splitting image: {e}")
        raise

# ------------------------------
# Inference Functions
# ------------------------------

def default_transform(Config):
    """
    Defines the transformation to be applied to each image tile before inference.

    Args:
        Config: Configuration object containing image_size.

    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    transform = transforms.Compose([
        transforms.Resize(Config.image_size),
        transforms.ToTensor()
    ])
    return transform

def run_inference_save_mask(Config, model, image_path, device, output_dir="./output/tiles/masks"):
    """
    Run inference on a single image tile and save the predicted mask.

    Args:
        Config: Configuration object.
        model (torch.nn.Module): The trained model.
        image_path (str): Path to the input image tile.
        device (torch.device): Device to run the inference on (CPU/GPU).
        output_dir (str): Directory to save the predicted mask.

    Returns:
        str: Path to the saved mask image.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Tile image not found at {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    transform = default_transform(Config)  # Use the default transform here

    try:
        model.eval()
        with torch.no_grad():
            # Load and transform the image
            with Image.open(image_path) as image:
                image = image.convert("L")  # Ensure grayscale
                input_tensor = transform(image).unsqueeze(0).to(device)

            # Perform inference
            output = model(input_tensor)

            # Assuming output is a single-channel mask
            output_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)  # Binarize the output

            # Scale mask to 0-255
            mask_array = output_mask * 255
            mask_image = Image.fromarray(mask_array)

            # Save the mask with tile indices
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            mask_image.save(mask_path)
            #print(f"Mask saved to {mask_path}")

            return mask_path

    except Exception as e:
        print(f"Error during inference on {image_path}: {e}")
        raise

def run_inference_on_tiles(Config, model, device, tiles_dir='./output/tiles/images', masks_output_dir="./output/tiles/masks", max_workers=4):
    """
    Runs inference on all tiles in the specified directory.

    Args:
        Config: Configuration object.
        model (torch.nn.Module): The trained model.
        device (torch.device): Device to run the inference on (CPU/GPU).
        tiles_dir (str): Directory containing the tile images.
        masks_output_dir (str): Directory to save the predicted masks.
        max_workers (int): Maximum number of parallel workers (default is 4).

    Returns:
        list: List of file paths to the saved mask images.
    """
    if not os.path.isdir(tiles_dir):
        raise FileNotFoundError(f"Tiles directory not found at {tiles_dir}")

    # Supported image formats
    supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

    # List and sort tile files
    tile_files = sorted([
        f for f in os.listdir(tiles_dir) 
        if f.lower().endswith(supported_formats)
    ])

    if not tile_files:
        raise ValueError(f"No tile images found in {tiles_dir} with formats {supported_formats}")

    os.makedirs(masks_output_dir, exist_ok=True)

    mask_paths = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_tile = {
            executor.submit(run_inference_save_mask, Config, model, os.path.join(tiles_dir, tile_file), device, masks_output_dir): tile_file
            for tile_file in tile_files
        }

        for future in as_completed(future_to_tile):
            tile_file = future_to_tile[future]
            try:
                mask_path = future.result()
                mask_paths.append(mask_path)
            except Exception as e:
                print(f"Failed to process tile {tile_file}: {e}")

    return mask_paths

# ------------------------------
# Combining Function with Blending
# ------------------------------

def combine_tiles_into_image_with_blending(original_image_path, masks_dir='./output/tiles/masks', output_image_path='./output/tiles/combined_mask.png', tile_size=512, overlap=50):
    """
    Combines mask tiles into a single large mask image with blending in overlapping regions.

    Args:
        masks_dir (str): Directory containing the mask tiles.
        output_image_path (str): Path to save the combined mask image.
        original_image_path (str): Path to the original large image to determine size.
        tile_size (int): Size of each tile (default is 512).
        overlap (int): Number of overlapping pixels between tiles (default is 50).
    """
    if not os.path.isfile(original_image_path):
        raise FileNotFoundError(f"Original image not found at {original_image_path}")

    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Masks directory not found at {masks_dir}")

    try:
        with Image.open(original_image_path) as original_image:
            original_image = original_image.convert("L")
            orig_width, orig_height = original_image.size

        # List and sort mask files
        supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        mask_files = sorted([
            f for f in os.listdir(masks_dir)
            if f.lower().endswith(supported_formats) and '_mask' in f
        ])

        # Calculate expected number of tiles
        stride = tile_size - overlap
        num_tiles_x = (orig_width - overlap) // stride
        num_tiles_y = (orig_height - overlap) // stride
        expected_num_tiles = num_tiles_x * num_tiles_y

        if len(mask_files) != expected_num_tiles:
            raise ValueError(f"Expected {expected_num_tiles} mask tiles, but found {len(mask_files)}.")

        # Initialize arrays for the combined mask and weight map
        combined_mask = np.zeros((orig_height, orig_width), dtype=np.float32)
        weight_map = np.zeros((orig_height, orig_width), dtype=np.float32)

        for mask_file in mask_files:
            try:
                # Extract tile indices from filename
                # Expected format: {base_name}_tile_{i}_{j}_mask.png
                parts = mask_file.replace('.png', '').replace('.jpg', '').replace('.jpeg', '').split('_')
                if len(parts) < 5:
                    raise ValueError(f"Mask filename {mask_file} does not conform to the expected format.")
                i = int(parts[-3])  # Assuming format: base_name_tile_i_j_mask.png
                j = int(parts[-2])

                left = j * stride
                upper = i * stride
                right = left + tile_size
                lower = upper + tile_size

                # Handle edge cases
                right = min(right, orig_width)
                lower = min(lower, orig_height)
                left = max(right - tile_size, 0)
                upper = max(lower - tile_size, 0)

                mask_path = os.path.join(masks_dir, mask_file)
                with Image.open(mask_path) as mask:
                    mask = mask.convert("L")
                    mask_array = np.array(mask, dtype=np.float32) / 255.0  # Normalize to [0,1]

                # Define blending weights
                weight = np.ones((mask_array.shape[0], mask_array.shape[1]), dtype=np.float32)

                # Left overlap
                if left > 0 and overlap > 0:
                    overlap_region = slice(0, overlap)
                    weight[:, overlap_region] *= np.linspace(0, 1, overlap)[None, :]

                # Right overlap
                if right < orig_width and overlap > 0:
                    overlap_region = slice(-overlap, None)
                    weight[:, overlap_region] *= np.linspace(1, 0, overlap)[None, :]

                # Top overlap
                if upper > 0 and overlap > 0:
                    overlap_region = slice(0, overlap)
                    weight[overlap_region, :] *= np.linspace(0, 1, overlap)[:, None]

                # Bottom overlap
                if lower < orig_height and overlap > 0:
                    overlap_region = slice(-overlap, None)
                    weight[overlap_region, :] *= np.linspace(1, 0, overlap)[:, None]

                # Add to combined mask and weight map
                combined_mask[upper:lower, left:right] += mask_array * weight
                weight_map[upper:lower, left:right] += weight

                #print(f"Blended {mask_file} at position ({left}, {upper})")

            except Exception as e:
                print(f"Error processing mask {mask_file}: {e}")
                raise

        # Avoid division by zero
        weight_map[weight_map == 0] = 1.0
        final_mask = combined_mask / weight_map
        final_mask = (final_mask * 255).astype(np.uint8)

        # Convert to PIL Image and save
        final_mask_image = Image.fromarray(final_mask)
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        final_mask_image.save(output_image_path)
        print(f"Combined mask with blending saved to {output_image_path}")

        # Optional: Validate the combined mask by checking dimensions
        if final_mask_image.size != (orig_width, orig_height):
            print("Warning: Combined mask size does not match original image size.")
        else:
            print("Combined mask size matches the original image.")

    except Exception as e:
        print(f"Error while combining tiles with blending: {e}")
        raise

# ------------------------------
# Main Processing Function
# ------------------------------

def run_main_tiles(Config, image_dir, output_dir, model, device, tile_size=512, overlap=50, max_workers=4):
    """
    Processes all images in the specified directory: splits into tiles, runs inference, and combines masks.

    Args:
        Config: Configuration object.
        image_dir (str): Directory containing input images.
        output_dir (str): Directory for base outputs
        model (torch.nn.Module): The trained model.
        device (torch.device): Device to run the inference on (CPU/GPU).
        tile_size (int): Size of each tile (default: 512).
        overlap (int): Overlap between tiles in pixels (default: 50).
        max_workers (int): Maximum number of parallel workers for inference (default: 4).
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

        # Define output directories for this image
        image_output_dir = os.path.join(output_dir, base_name, 'tiles', 'images')
        masks_output_dir = os.path.join(output_dir, base_name, 'tiles', 'masks')
        combined_mask_dir = os.path.join(output_dir, base_name, 'combined_masks')
        combined_mask_path = os.path.join(combined_mask_dir, f"{base_name}_combined_mask.png")

        try:
            # Step 1: Split image into tiles
            split_image_into_tiles(
                image_path=image_path,
                tile_size=tile_size,
                overlap=overlap,
                output_dir=image_output_dir
            )

            # Step 2: Run inference on tiles
            run_inference_on_tiles(
                Config=Config,
                model=model,
                device=device,
                tiles_dir=image_output_dir,
                masks_output_dir=masks_output_dir,
                max_workers=max_workers
            )

            # Step 3: Combine masks with blending
            combine_tiles_into_image_with_blending(
                original_image_path=image_path,
                masks_dir=masks_output_dir,
                output_image_path=combined_mask_path,
                tile_size=tile_size,
                overlap=overlap
            )

            print(f"Finished processing image: {image_path}")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Optionally, you can continue with the next image or halt execution
            continue

# ------------------------------
# Main Execution Block
# ------------------------------

if __name__ == '__main__':
    print("main")