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
from config.config import Config 

# ------------------------------
# Splitting Function
# ------------------------------

def split_image_into_tiles(image_path, tile_size=512, overlap=50, save_tiles=False, output_dir='./output/tiles/images'):
    """
    Splits a large image into smaller tiles with optional overlaps and optional saving to disk.

    Args:
        image_path (str): Path to the input image.
        tile_size (int): Size of each tile (default is 512).
        overlap (int): Number of pixels to overlap between tiles (default is 50).
        save_tiles (bool): Whether to save the tiles to disk (default is False).
        output_dir (str): Directory to save the split tiles if saving is enabled.

    Returns:
        list of dict: Each dict contains 'tile' (PIL Image) and 'position' (tuple).
                      If save_tiles is True, includes 'tile_path'.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found at {image_path}")

    tiles = []

    try:
        with Image.open(image_path) as img:
            img = img.convert("L")  # Convert to grayscale if needed
            width, height = img.size

            # Validate image size
            if width < tile_size or height < tile_size:
                raise ValueError(f"Image size should be at least {tile_size}x{tile_size} pixels.")

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

                    # Handle edge cases
                    if right > width:
                        right = width
                        left = max(width - tile_size, 0)
                    if lower > height:
                        lower = height
                        upper = max(height - tile_size, 0)

                    tile = img.crop((left, upper, right, lower))

                    # Optionally save the tile to disk
                    if save_tiles:
                        os.makedirs(output_dir, exist_ok=True)
                        base_name = os.path.splitext(os.path.basename(image_path))[0]
                        tile_path = os.path.join(output_dir, f"{base_name}_tile_{i}_{j}.png")
                        tile.save(tile_path)
                        tiles.append({
                            'tile': tile,
                            'position': (left, upper, right, lower),
                            'tile_path': tile_path  # Include path if saved
                        })
                    else:
                        tiles.append({
                            'tile': tile,
                            'position': (left, upper, right, lower)
                        })

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

def run_inference(Config, model, tile, device):
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
            # Define the transform
            transform = default_transform(Config)

            # Apply transformations
            input_tensor = transform(tile).unsqueeze(0).to(device)

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

def run_inference_on_tiles(Config, model, device, tiles, save_masks=False, masks_output_dir="./output/tiles/masks", max_workers=4):
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
        mask = run_inference(Config, model, tile, device)

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
            mask_image = Image.fromarray(mask)
            mask_image.save(mask_path)
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
        with Image.open(original_image_path) as original_image:
            original_image = original_image.convert("L")
            orig_width, orig_height = original_image.size

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
        final_mask_image = Image.fromarray(final_mask)
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        final_mask_image.save(output_image_path)
        print(f"Combined mask with blending saved to {output_image_path}")

        # Optional: Validate the combined mask by checking dimensions
        if final_mask_image.size != (orig_width, orig_height):
            print("Warning: Combined mask size does not match original image size.")
        else:
            print("Combined mask size matches the original image.")

        # Optional: Combine original image and mask side by side
        try:
            with Image.open(original_image_path) as original_image:
                original_image = original_image.convert("L")
                # Resize original image to match mask if necessary
                if original_image.size != final_mask_image.size:
                    original_image = original_image.resize(final_mask_image.size, Image.LANCZOS)
                original_array = np.array(original_image)
                mask_array = final_mask  # Already a NumPy array

                # Combine original image and mask side by side
                combined_side_by_side = np.hstack((original_array, mask_array))

                # Convert to PIL Image
                combined_image = Image.fromarray(combined_side_by_side)

                # Define the path for the combined image
                combined_image_path = os.path.join(
                    os.path.dirname(output_image_path),
                    f"combined_{os.path.basename(original_image_path)}"
                )

                # Save the combined image
                combined_image.save(combined_image_path)
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

def run_main_tiles(Config, image_dir, output_dir, model, device, tile_size=512, overlap=50, max_workers=4, save_tiles=False, save_masks=False):
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
                max_workers=max_workers
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

# ------------------------------
# Main Execution Block
# ------------------------------

if __name__ == '__main__':
    print("Starting in-memory tile processing pipeline...")

    # Example usage:
    # Assume Config is properly defined in config.config
    # model is loaded and moved to the appropriate device
    # image_dir and output_dir are specified

    # Example placeholders (replace with actual implementations)
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load your trained model here
    # Example:
    # from your_model_module import YourModelClass
    # model = YourModelClass()
    # model.load_state_dict(torch.load('path_to_model.pth'))
    # model.to(device)

    # Ensure that 'model' is defined. Replace with actual model loading code.
    try:
        model
    except NameError:
        raise NameError("Model is not defined. Please load your trained model before running the pipeline.")

    image_directory = '/path/to/input/images'  # Replace with your input image directory
    output_directory = '/path/to/output'       # Replace with your desired output directory

    # Run the main processing function
    run_main_tiles(
        Config=config,
        image_dir=image_directory,
        output_dir=output_directory,
        model=model,
        device=device,
        tile_size=512,
        overlap=50,
        max_workers=4
    )
