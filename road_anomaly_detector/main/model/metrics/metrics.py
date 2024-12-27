import os
import csv
import time
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# Albumentations imports
import albumentations as A
from albumentations.pytorch import ToTensorV2

from numba import jit
from sklearn.metrics import f1_score
from scipy.spatial import KDTree
from config.config import Config

###############################################################################
# Helper Functions
###############################################################################
def binarize_output(output, threshold=0.5):
    return (output > threshold).astype(np.uint8)

@jit(nopython=True)
def compute_metrics(predictions, ground_truths):
    # Ensure both arrays are binary (0 or 1)
    predictions = (predictions > 0.5).astype(np.uint8)
    ground_truths = (ground_truths > 0.5).astype(np.uint8)

    # Compute true positives, false positives, false negatives
    TP_d = np.sum(predictions & ground_truths)
    FP_d = np.sum(predictions & ~ground_truths)
    FN_d = np.sum(~predictions & ground_truths)

    correctness = TP_d / (TP_d + FP_d) if (TP_d + FP_d) > 0 else 0
    completeness = TP_d / (TP_d + FN_d) if (TP_d + FN_d) > 0 else 0
    quality = TP_d / (TP_d + FP_d + FN_d) if (TP_d + FP_d + FN_d) > 0 else 0

    return correctness, completeness, quality

def hausdorff_distance_95(prediction, ground_truth):
    prediction = prediction.reshape(-1, 2)
    ground_truth = ground_truth.reshape(-1, 2)
    tree_pred = KDTree(prediction)
    tree_gt = KDTree(ground_truth)
    distances_pred_to_gt, _ = tree_gt.query(prediction, k=1, workers=-1)
    distances_gt_to_pred, _ = tree_pred.query(ground_truth, k=1, workers=-1)
    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    hd95 = np.percentile(all_distances, 95)
    return hd95

###############################################################################
# Albumentations Transform
###############################################################################
def default_transform(Config):
    """
    Returns an Albumentations Compose transform.
    Adjust according to your needs (normalization, etc.).
    """
    transform = A.Compose([
        A.Resize(height=Config.image_size[0], width=Config.image_size[1]),
        # A.Normalize(mean=(mean,), std=(std,)),  # optionally normalize
        ToTensorV2()
    ])
    return transform

###############################################################################
# Model Evaluation
###############################################################################
def evaluate_model(Config, model, test_loader, device):
    model.eval()
    batch_metrics = {
        "correctness": [],
        "completeness": [],
        "quality": [],
        "f1": []
        # "hd95": []
    }

    total_batches = len(test_loader)
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            # # Display the image and mask
            # # Assuming batch size > 1, pick the first image and mask in the batch
            # image = images[0].cpu().numpy().squeeze()  # Convert to numpy and remove extra dimensions
            # mask = masks[0].cpu().numpy().squeeze()    # Convert to numpy and remove extra dimensions
            
            # # Plot the image and its corresponding mask
            # plt.figure(figsize=(12, 6))
            
            # # Plot the image
            # plt.subplot(1, 2, 1)
            # plt.imshow(image, cmap="gray")
            # plt.title("Image")
            # plt.axis("off")
            
            # # Plot the mask
            # plt.subplot(1, 2, 2)
            # plt.imshow(mask, cmap="gray")
            # plt.title("Mask")
            # plt.axis("off")
            
            # # Display the plot
            # print("Press any key to close the plot and continue...")
            # plt.show(block=True)
            
            outputs = (model(images) > 0.5).float().cpu().numpy()
            masks = masks.cpu().numpy()
            print("done with inference")

            # Accumulate batch metrics
            batch_correctness = []
            batch_completeness = []
            batch_quality = []
            batch_f1 = []
            # batch_hd95 = []
            
            for pred, gt in zip(outputs, masks):
                gt_binary = (gt > 0.5).astype(np.uint8)
                correctness, completeness, quality = compute_metrics(pred, gt_binary)
                batch_correctness.append(correctness)
                batch_completeness.append(completeness)
                batch_quality.append(quality)
                batch_f1.append(f1_score(gt_binary.flatten(), pred.flatten()))
                # batch_hd95.append(hausdorff_distance_95(pred, gt_binary))

            # Calculate the average for the batch
            batch_metrics["correctness"].append(np.mean(batch_correctness))
            batch_metrics["completeness"].append(np.mean(batch_completeness))
            batch_metrics["quality"].append(np.mean(batch_quality))
            batch_metrics["f1"].append(np.mean(batch_f1))
            # batch_metrics["hd95"].append(np.mean(batch_hd95))

            if i % max(1, total_batches // 10) == 0 or i == total_batches - 1:
                print(f"Processed {i + 1}/{total_batches} batches "
                      f"({(i + 1) / total_batches * 100:.2f}%)")
        
    # Write metrics to CSV
    with open(Config.metric_save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Average"])  # Header row

        print("\nTest Set Metrics:")
        for metric, values in batch_metrics.items():
            avg_value = np.mean(values)
            print(f"{metric.capitalize()}: {avg_value:.4f}")
            writer.writerow([metric.capitalize(), f"{avg_value:.4f}"])

    print(f"\nMetrics saved to {Config.metric_save_path}")

###############################################################################
# Inference on a Single Image
###############################################################################
def run_inference(Config, model, image_path, device, output_dir="./outputs", preprocessing=False):
    """
    Run inference on a single image and save the predicted mask using OpenCV + Albumentations.
    """
    model.eval()
    transform = default_transform(Config)

    with torch.no_grad():
        # 1) Load image (grayscale) with OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        if preprocessing==True:
            image = apply_preprocessing(image)

        # 2) Apply Albumentations transform
        # Albumentations expects a dict {"image": <numpy array>}
        # which returns {"image": <torch.Tensor>}
        transformed = transform(image=image)
        input_tensor = transformed["image"].unsqueeze(0).to(device)
        input_tensor = input_tensor.float().to(device) / 255

        # 3) Perform inference
        output = model(input_tensor)
        output_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        # 4) Resize the original image to match the predicted mask dimensions
        #    (mask is [H, W], so shape is (height, width))
        resized_image = cv2.resize(
            image,
            (output_mask.shape[1], output_mask.shape[0]),
            interpolation=cv2.INTER_LANCZOS4
        )

        # 5) Combine resized input image (left) and predicted mask (right)
        mask_array = output_mask * 255
        combined_image = np.hstack((resized_image, mask_array))

        # 6) Save the combined image
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(image_path)
        combined_path = os.path.join(output_dir, f"combined_{base_name}")
        cv2.imwrite(combined_path, combined_image)
        print(f"Combined image saved to {combined_path}")

        # 7) (Optional) Visualize using matplotlib
        plt.figure(figsize=(12, 6))
        plt.title("Input Image and Predicted Mask")
        plt.imshow(combined_image, cmap="gray")
        plt.axis("off")
        plt.show()

###############################################################################
# Inference on a Folder of Images
###############################################################################
def run_inference_on_folder(Config, model, folder_path, device, output_dir="./outputs", preprocessing=False):
    """
    Run inference on all images in a folder and save the predicted masks using OpenCV + Albumentations.
    """
    model.eval()
    transform = default_transform(Config)
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all files in the folder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        # Optionally check the extension
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            with torch.no_grad():
                # 1) Read image with OpenCV
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Skipping {image_path}, cannot read.")
                    continue

                if preprocessing==True:
                    image = apply_preprocessing(image)

                # 2) Transform -> Torch Tensor
                transformed = transform(image=image)
                input_tensor = transformed["image"].unsqueeze(0).to(device)
                input_tensor = input_tensor.float().to(device) / 255

                # 3) Inference
                output = model(input_tensor)
                output_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

                # 4) Resize the original image to match the predicted mask dimensions
                resized_image = cv2.resize(
                    image,
                    (output_mask.shape[1], output_mask.shape[0]),
                    interpolation=cv2.INTER_LANCZOS4
                )

                # 5) Combine them side by side
                mask_array = output_mask * 255
                combined_image = np.hstack((resized_image, mask_array))

                # 6) Save combined image
                combined_path = os.path.join(output_dir, f"combined_{image_name}")
                cv2.imwrite(combined_path, combined_image)
                print(f"Combined image saved to {combined_path}")

                # 7) (Optional) visualize
                # plt.figure(figsize=(12, 6))
                # plt.title(f"Input Image and Predicted Mask - {image_name}")
                # plt.imshow(combined_image, cmap="gray")
                # plt.axis("off")
                # plt.show()


def apply_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE and Bilateral Filtering to a grayscale image.

    Parameters:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: The preprocessed image.
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image with 2 dimensions.")

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=45, tileGridSize=(10, 10))
    clahe_img = clahe.apply(image)

    # Apply Bilateral Filter
    # Bilateral filtering preserves edges while smoothing
    bilateral_filtered = cv2.bilateralFilter(
        clahe_img, 
        d=20,            # Diameter of each pixel neighborhood
        sigmaColor=26,   # Filter sigma in the color space
        sigmaSpace=5     # Filter sigma in the coordinate space
    )

    return bilateral_filtered
