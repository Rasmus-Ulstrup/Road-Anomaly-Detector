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


def binarize_output(output, threshold=0.5):
    return (output > threshold).astype(np.uint8)

def compute_metrics(predictions, ground_truths):
    # Ensure both arrays are binary (0 or 1)
    predictions = (predictions > 0.5).astype(np.uint8)
    ground_truths = (ground_truths > 0.5).astype(np.uint8)

    # Compute true positives, false positives, false negatives
    TP_d = np.sum(predictions & ground_truths)
    FP_d = np.sum(predictions & ~ground_truths)
    FN_d = np.sum(~predictions & ground_truths)

    # Check if there are any positive samples in both y_true and y_pred
    has_positive_true = np.any(ground_truths == 1)
    has_positive_pred = np.any(predictions == 1)

    if not has_positive_true and not has_positive_pred:
        # No positive samples in both y_true and y_pred
        correctness = 1.0
        completeness = 1.0
        quality = 1.0
    else:
        correctness = TP_d / (TP_d + FP_d) if (TP_d + FP_d) > 0 else 0
        completeness = TP_d / (TP_d + FN_d) if (TP_d + FN_d) > 0 else 0
        quality = TP_d / (TP_d + FP_d + FN_d) if (TP_d + FP_d + FN_d) > 0 else 0

    return correctness, completeness, quality

def compute_centerline_metrics(pred_mask, ref_mask, buffer_radius=2, visualize=True):
    """
    Compute centerline metrics between predicted and reference masks with optional visualization.

    Parameters:
    - pred_mask (numpy.ndarray): Predicted mask.
    - ref_mask (numpy.ndarray): Reference mask.
    - buffer_radius (int): Radius for dilation.
    - visualize (bool): If True, display visualizations.

    Returns:
    - correctness (float): TP / (TP + FP)
    - completeness (float): TP / (TP + FN)
    - quality (float): TP / (TP + FP + FN)
    """

    # 1) Convert to binary uint8
    pred_bin = (pred_mask > 0.5).astype(np.uint8) * 255
    if pred_bin.ndim == 3:
        pred_bin = np.squeeze(pred_bin, axis=0)
    ref_bin = (ref_mask > 0.5).astype(np.uint8) * 255
    if ref_bin.ndim == 3:
        ref_bin = np.squeeze(ref_bin, axis=0)

    if visualize:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Predicted Binary Mask')
        plt.imshow(pred_bin, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Reference Binary Mask')
        plt.imshow(ref_bin, cmap='gray')
        plt.axis('off')
        plt.show()

    # 2) Apply Zhang–Suen thinning (centerline extraction)
    # Make sure cv2.ximgproc exists in your environment (opencv-contrib-python).
    pred_thin = cv2.ximgproc.thinning(pred_bin, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    ref_thin  = cv2.ximgproc.thinning(ref_bin,  thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    if visualize:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Thinned Predicted Mask')
        plt.imshow(pred_thin, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Thinned Reference Mask')
        plt.imshow(ref_thin, cmap='gray')
        plt.axis('off')
        plt.show()

    # Count the total skeleton length (simply the number of nonzero “thin” pixels)
    len_pred = np.count_nonzero(pred_thin)
    len_ref  = np.count_nonzero(ref_thin)

    # Edge case: if both are empty, consider it a perfect match
    if len_pred == 0 and len_ref == 0:
        if visualize:
            print("Both predicted and reference masks are empty. Perfect match.")
        return 1.0, 1.0, 1.0

    # 3) Create a disk/ellipse structuring element for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*buffer_radius + 1, 2*buffer_radius + 1))

    # --- Step 1: Buffer around REF, see what portion of PRED is inside
    ref_dilated = cv2.dilate(ref_thin, kernel)
    matched_pred = (pred_thin > 0) & (ref_dilated > 0)
    TP_d = np.count_nonzero(matched_pred)
    FP_d = len_pred - TP_d

    # --- Step 2: Buffer around PRED, see what portion of REF is inside
    pred_dilated = cv2.dilate(pred_thin, kernel)
    matched_ref = (ref_thin > 0) & (pred_dilated > 0)
    matched_ref_len = np.count_nonzero(matched_ref)
    FN_d = len_ref - matched_ref_len

    if visualize:
        # Create color overlays
        overlay_pred = np.zeros((*pred_bin.shape, 3), dtype=np.uint8)
        overlay_ref = np.zeros((*ref_bin.shape, 3), dtype=np.uint8)

        # True Positives: Green
        overlay_pred[matched_pred] = [0, 255, 0]

        # False Positives: Red
        overlay_pred[pred_thin > 0] &= 0  # Reset
        overlay_pred[matched_pred] = [0, 255, 0]
        overlay_pred[pred_thin > 0] = [255, 0, 0]  # Red for all pred_thin
        overlay_pred[matched_pred] = [0, 255, 0]  # Override TP to green

        # False Negatives: Blue
        overlay_ref[matched_ref] = [0, 255, 0]
        overlay_ref[ref_thin > 0] = [0, 0, 255]  # Blue for all ref_thin
        overlay_ref[matched_ref] = [0, 255, 0]  # Override TP to green

        # Plotting
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.title('Predicted Binary Mask')
        plt.imshow(pred_bin, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.title('Reference Binary Mask')
        plt.imshow(ref_bin, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.title('Thinned Masks')
        combined_thin = np.zeros((*pred_bin.shape, 3), dtype=np.uint8)
        combined_thin[..., 0] = pred_thin  # Red channel for pred_thin
        combined_thin[..., 1] = ref_thin   # Green channel for ref_thin
        plt.imshow(combined_thin)
        plt.axis('off')
        plt.legend(['Predicted Thin', 'Reference Thin'])

        plt.subplot(2, 3, 4)
        plt.title('Dilated Reference Mask')
        plt.imshow(ref_dilated, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.title('Dilated Predicted Mask')
        plt.imshow(pred_dilated, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.title('Matched Areas')
        combined_overlay = cv2.addWeighted(overlay_pred, 1, overlay_ref, 1, 0)
        plt.imshow(combined_overlay)
        plt.axis('off')
        plt.legend(['True Positives', 'False Positives', 'False Negatives'])

        plt.tight_layout()
        plt.show()

    # 4) Compute metrics
    # correctness = TP_d / (TP_d + FP_d)
    denom_correctness = (TP_d + FP_d)
    correctness = TP_d / denom_correctness if denom_correctness > 0 else 0.0

    # completeness = TP_d / (TP_d + FN_d)
    denom_completeness = (TP_d + FN_d)
    completeness = TP_d / denom_completeness if denom_completeness > 0 else 0.0

    # quality = TP_d / (TP_d + FP_d + FN_d)
    denom_quality = (TP_d + FP_d + FN_d)
    quality = TP_d / denom_quality if denom_quality > 0 else 0.0

    if visualize:
        print(f"Metrics:\nCorrectness: {correctness:.4f}\nCompleteness: {completeness:.4f}\nQuality: {quality:.4f}")

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


def default_transform(Config):
    """
    Returns an Albumentations Compose transform.
    Adjust according to your needs (normalization, etc.).
    """
    transform = A.Compose([
        A.Resize(
            height=Config.image_size[0],
            width=Config.image_size[1],
            interpolation=cv2.INTER_LINEAR,         # For continuous image
            mask_interpolation=cv2.INTER_NEAREST    # For discrete mask
        ),
        ToTensorV2()
    ])
    return transform


def evaluate_model(Config, model, test_loader, device):
    model.eval()
    batch_metrics = {
        "correctness": [],
        "completeness": [],
        "quality": [],
        "precision": [],
        "recall": [],
        "IoU": [],
        "f1": []
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
            
            # Display the plot
            # print("Press any key to close the plot and continue...")
            # plt.show(block=True)
            
            outputs = model(images)  # Expecting a list
            if not isinstance(outputs, list):
                final_outputs = (outputs.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            else:
                final_output = outputs[-1]
                # Extract the final output
                final_outputs = (final_output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            masks = masks.cpu().numpy()
            print("done with inference")
            # print(np.unique(mask))
            # Accumulate batch metrics
            batch_correctness = []
            batch_completeness = []
            batch_quality = []
            batch_precision = []
            batch_recall = []
            batch_IoU = []
            batch_f1 = []
            # batch_hd95 = []
            
            for pred, gt in zip(final_outputs, masks):
                gt_binary = (gt > 0.5).astype(np.uint8)
                precision, recall, IoU = compute_metrics(pred, gt_binary)
                batch_precision.append(precision)
                batch_recall.append(recall)
                batch_IoU.append(IoU)

                correctness, completeness, quality = compute_centerline_metrics(pred, gt_binary, 2)
                batch_correctness.append(correctness)
                batch_completeness.append(completeness)
                batch_quality.append(quality)
                batch_f1.append(f1_score(gt_binary.flatten(), pred.flatten()))
                # batch_hd95.append(hausdorff_distance_95(pred, gt_binary))

            # Calculate the average for the batch
            batch_metrics["correctness"].append(np.mean(batch_correctness))
            batch_metrics["completeness"].append(np.mean(batch_completeness))
            batch_metrics["quality"].append(np.mean(batch_quality))
            batch_metrics["precision"].append(np.mean(batch_precision))
            batch_metrics["recall"].append(np.mean(batch_recall))
            batch_metrics["IoU"].append(np.mean(batch_IoU))
            batch_metrics["f1"].append(np.mean(batch_f1))

            # print("Current Batch Metrics:")
            # print(f"Correctness: {np.mean(batch_correctness):.4f}")
            # print(f"Completeness: {np.mean(batch_completeness):.4f}")
            # print(f"Quality: {np.mean(batch_quality):.4f}")
            # print(f"Precision: {np.mean(batch_precision):.4f}")
            # print(f"Recall: {np.mean(batch_recall):.4f}")
            # print(f"IoU: {np.mean(batch_IoU):.4f}")
            # print(f"F1: {np.mean(batch_f1):.4f}")
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
        # output = model(input_tensor)
        # output_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        #BEAR
        outputs = model(input_tensor)  # Expecting a list
        if not isinstance(outputs, list):
            output_mask = (outputs.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        else:
            final_output = outputs[-1]
            # Extract the final output
            output_mask = (final_output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

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
                # output = model(input_tensor)
                # output_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                #BEAR
                outputs = model(input_tensor)  # Expecting a list
                if not isinstance(outputs, list):
                    output_mask = (outputs.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                else:
                    final_output = outputs[-1]
                    # Extract the final output
                    output_mask = (final_output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

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
