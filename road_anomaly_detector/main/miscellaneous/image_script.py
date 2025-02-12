import os
import shutil
import string

def copy_images(source_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    combined_prefix = "combined_"
    image_extension = ".png"

    matching_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.startswith(combined_prefix) and file.lower().endswith(image_extension):
                full_path = os.path.join(root, file)
                matching_files.append(full_path)

    if not matching_files:
        print("No matching files found to copy.")
        return

    for file_path in matching_files:
        filename = os.path.basename(file_path)
        new_filename = filename[len(combined_prefix):]  # Remove 'combined_'
        target_path = os.path.join(output_dir, new_filename)

        try:
            shutil.copy2(file_path, target_path)
            print(f"Copied: {file_path} -> {target_path}")
        except Exception as e:
            print(f"Failed to copy {file_path} to {target_path}: {e}")

    print("\nImage copying completed.")

if __name__ == "__main__":
    # Define the source and base output directories
    source_directory = r"/home/crack/Road-Anomaly-Detector/road_anomaly_detector/main/model/output/tiles"
    base_output_directory = r"/home/crack/Road-Anomaly-Detector/road_anomaly_detector/main/miscellaneous/tiles_folder"

    # Prom
    subfolder_name = input("Please enter the name of the output subfolder: \n").strip()

    #
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    if not subfolder_name:
        raise ValueError("Subfolder name cannot be empty.")
    if not all(c in valid_chars for c in subfolder_name):
        raise ValueError("Invalid folder name. Please avoid using special characters.")

    # 
    output_directory = os.path.join(base_output_directory, subfolder_name)

    # copy images
    copy_images(source_directory, output_directory)
