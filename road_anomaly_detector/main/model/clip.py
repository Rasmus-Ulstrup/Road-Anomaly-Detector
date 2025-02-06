from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import os
from utils.tiles import split_image_into_tiles, combine_tiles_into_image_with_blending
import numpy as np

def load_model():
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    return model, processor

def run_inference(model, processor, image_path, prompts):
    new_input_size = (512, 512)

    # Load and process the image
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor.image_processor(images=image, return_tensors="pt", size=new_input_size)

    # Repeat the image
    batch_size = len(prompts)
    pixel_values = image_inputs["pixel_values"].repeat(batch_size, 1, 1, 1)

    text_inputs = processor.tokenizer(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Transfer model and input to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Transfer model to GPU
    pixel_values = pixel_values.to(device)
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    # Combine the processed inputs
    inputs = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    # Forward pass on GPU
    outputs = model(**inputs, interpolate_pos_encoding=True)
    return outputs, image

def run_inference_tiles(model, processor, image_path, working_dir, prompts, overlap=0):
    # Define directories
    tiles_dir = os.path.join(working_dir, 'images')
    masks_dir = os.path.join(working_dir, 'mask')
    combined_mask_dir = os.path.join(working_dir, 'combined_mask')
    combined_mask_path = os.path.join(combined_mask_dir, 'combined_mask.png')
    
    # Step 1: Split the image into tiles
    split_image_into_tiles(
        image_path=image_path,
        tile_size=512,
        overlap=overlap,
        output_dir=tiles_dir
    )
    
    # Step 2: Ensure the masks directory exists
    os.makedirs(masks_dir, exist_ok=True)
    
    # Step 3: List all tile images
    tile_files = sorted([
        f for f in os.listdir(tiles_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))
    ])
    
    if not tile_files:
        raise ValueError(f"No tile images found in {tiles_dir}.")
    
    # Step 4: Process each tile
    for tile_file in tile_files:
        tile_path = os.path.join(tiles_dir, tile_file)
        
        # Run inference on the tile
        outputs, tile_image = run_inference(model, processor, tile_path, prompts)
        
        # Extract the segmentation mask
        segmentation_masks = outputs.logits  # Shape: (batch_size, height, width)
        
        # Assuming single prompt; adjust if multiple prompts are used
        index_to_visualize = 0
        segmentation_mask = segmentation_masks[index_to_visualize]
        
        # Apply sigmoid to get probabilities
        segmentation_mask = segmentation_mask.sigmoid()
        binary_mask = (segmentation_mask > 0.55).float()
        binary_mask_np = binary_mask.cpu().numpy()
        
        segmentation_image = to_pil_image(binary_mask_np)
        
        # Define the mask path
        base_name = os.path.splitext(tile_file)[0]
        mask_filename = f"{base_name}_mask.png"
        mask_path = os.path.join(masks_dir, mask_filename)
        
        # Ensure the directory for mask_path exists
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        
        # Save the segmentation mask
        segmentation_image.save(mask_path)
        print(f"Saved mask: {mask_path}")
    
    # Step 5: Combine all masks into a single image with blending
    combine_tiles_into_image_with_blending(
        original_image_path=image_path,
        masks_dir=masks_dir,
        output_image_path=combined_mask_path,
        tile_size=512,
        overlap=overlap
    )
    
    print(f"Combined segmentation mask saved to: {combined_mask_path}")
    return combined_mask_path

def visualize(image, outputs, prompts):
    # Extract the segmentation mask
    segmentation_masks = outputs.logits  # Shape: (batch_size, height, width)

    # Choose a prompt index to visualize
    index_to_visualize = 0 
    segmentation_mask = segmentation_masks[index_to_visualize]

    # Normalize and convert to an image
    segmentation_mask = segmentation_mask.sigmoid()  # Apply sigmoid to get probabilities
    segmentation_image = to_pil_image(segmentation_mask)

    original_image = image  # Assuming `image` is the PIL.Image object before processing

    # Display the original image and segmentation mask side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis("on")

    # Display the segmentation mask
    ax[1].imshow(segmentation_image, cmap="viridis")  # Use a color map for better visualization
    ax[1].set_title(f"Segmentation Mask for Prompt: {prompts[index_to_visualize]}")
    ax[1].axis("on")

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    model, processor = load_model()

    prompts = ['crack in concrete']
    image_path = "more_models/clip/test_images/test3.png"
    working_dir = "more_models/clip/test_images/"
    
    # Run inference on tiles and get the combined mask path
    combined_mask_path = run_inference_tiles(model, processor, image_path, working_dir, prompts,overlap=256)
    
    # Load the original image and the combined mask
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    
    with Image.open(combined_mask_path) as combined_mask:
        outputs = type('Outputs', (object,), {'logits': torch.tensor(np.array(combined_mask), dtype=torch.float32).unsqueeze(0)})
    
    visualize(image, outputs, prompts)
