import numpy as np
from sklearn.metrics import f1_score
from PIL import Image

def calculate_f1_score(mask1, mask2, zero_division=1):
    """
    Calculate the F1 score between two binary masks.

    Parameters:
    - mask1: numpy array or filepath to the first mask (ground truth).
    - mask2: numpy array or filepath to the second mask (prediction).
    - zero_division: Value to return when precision or recall is undefined (default=1).

    Returns:
    - f1: F1 score as a float.
    """
    # Load masks if they are file paths
    if isinstance(mask1, str):
        mask1 = np.array(Image.open(mask1).convert("1"), dtype=np.uint8)
    if isinstance(mask2, str):
        mask2 = np.array(Image.open(mask2).convert("1"), dtype=np.uint8)

    # Check if masks are the same shape
    if mask1.shape != mask2.shape:
        raise ValueError(f"Masks must have the same dimensions, but got {mask1.shape} and {mask2.shape}.")

    # Flatten the masks
    mask1_flat = mask1.flatten()
    mask2_flat = mask2.flatten()

    # Calculate F1 score
    f1 = f1_score(mask1_flat, mask2_flat, zero_division=zero_division)

    return f1

# Example usage
if __name__ == "__main__":
    # Replace with paths to your mask files or numpy arrays
    mask1_path = "road_anomaly_detector/main/model/utils/vej_2_25_00004_mask.png"  # Path to ground truth mask
    mask2_path = "road_anomaly_detector/main/model/utils/vej_2_25_00004_image_combined_mask.png"    # Path to predicted mask

    # Calculate and print F1 score
    try:
        f1 = calculate_f1_score(mask1_path, mask2_path)
        print(f"F1 Score: {f1:.4f}")
    except ValueError as e:
        print(f"Error: {e}")
