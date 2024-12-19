import os
from PIL import Image

def stitch_images_from_directory(target_dir, output_dir):
    """
    Stitches all images in the target directory that share the same name (excluding incrementing parts).

    Args:
        target_dir (str): Path to the target directory containing the images.
        output_dir (str): Path to the directory to save stitched images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Group images by their base name
    grouped_images = {}
    for file in sorted(os.listdir(target_dir)):
        if file.endswith(".png"):
            base_name = "_".join(file.split("_")[:-1])
            if base_name not in grouped_images:
                grouped_images[base_name] = []
            grouped_images[base_name].append(os.path.join(target_dir, file))

    # Stitch images for each base name
    for base_name, image_paths in grouped_images.items():
        stitched_image = None
        for img_path in image_paths:
            img = Image.open(img_path)
            if stitched_image is None:
                stitched_image = img
            else:
                new_width = max(stitched_image.width, img.width)
                new_height = stitched_image.height + img.height
                new_image = Image.new("RGB", (new_width, new_height))
                new_image.paste(stitched_image, (0, 0))
                new_image.paste(img, (0, stitched_image.height))
                stitched_image = new_image

        # Save the stitched image
        output_path = os.path.join(output_dir, f"{base_name}_stitched.png")
        stitched_image.save(output_path)
        print(f"Stitched image saved at: {output_path}")

if __name__ == "__main__":
    # Define the base directories
    base_target_directory = r"/home/crack/Road-Anomaly-Detector/road_anomaly_detector/main/miscellaneous/tiles_folder"
    base_output_directory = r"/home/crack/Road-Anomaly-Detector/road_anomaly_detector/main/miscellaneous/stiched_folder"

    #Promt the user for the target subfolder nambe
    subfolder_name = input("Please enter the name of the target subfolder: \n").strip()

    # Construct the full target directory path
    target_directory = os.path.join(base_target_directory, subfolder_name)
    # Construct the full output directory path
    output_directory = os.path.join(base_output_directory, subfolder_name)

    # Stitch images
    stitch_images_from_directory(target_directory, output_directory)
