import numpy as np
import torch
from sklearn.metrics import f1_score
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import os
from PIL import Image  # Ensure this is imported at the top of the file

def binarize_output(output, threshold=0.5):
    return (output > threshold).astype(np.uint8)

def compute_metrics(predictions, ground_truths):
    predictions = predictions.flatten()
    ground_truths = ground_truths.flatten()

    TP_d = np.sum((predictions == 1) & (ground_truths == 1))
    FP_d = np.sum((predictions == 1) & (ground_truths == 0))
    FN_d = np.sum((predictions == 0) & (ground_truths == 1))

    correctness = TP_d / (TP_d + FP_d) if (TP_d + FP_d) > 0 else 0
    completeness = TP_d / (TP_d + FN_d) if (TP_d + FN_d) > 0 else 0
    quality = TP_d / (TP_d + FP_d + FN_d) if (TP_d + FP_d + FN_d) > 0 else 0

    return correctness, completeness, quality

def hausdorff_distance_95(prediction, ground_truth):
    prediction = prediction.reshape(-1, 2)
    ground_truth = ground_truth.reshape(-1, 2)
    hausdorff_pred_to_gt = directed_hausdorff(prediction, ground_truth)[0]
    hausdorff_gt_to_pred = directed_hausdorff(ground_truth, prediction)[0]
    hd95 = np.percentile([hausdorff_pred_to_gt, hausdorff_gt_to_pred], 95)
    return hd95

# Test evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    metrics = {"correctness": [], "completeness": [], "quality": [], "f1": [], "hd95": []}
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = (model(images) > 0.5).float().cpu().numpy()
            masks = masks.cpu().numpy()

            for pred, gt in zip(outputs, masks):
                pred_binary = (pred > 0.5).astype(np.uint8)
                gt_binary = (gt > 0.5).astype(np.uint8)

                correctness, completeness, quality = compute_metrics(pred_binary, gt_binary)
                metrics["correctness"].append(correctness)
                metrics["completeness"].append(completeness)
                metrics["quality"].append(quality)
                metrics["f1"].append(f1_score(gt_binary.flatten(), pred_binary.flatten()))
                metrics["hd95"].append(hausdorff_distance_95(pred_binary, gt_binary))

    # Print metrics
    print("\nTest Set Metrics:")
    for metric, values in metrics.items():
        print(f"{metric.capitalize()}: {np.mean(values):.4f}")

def run_inference(model, image_path, transform, device, output_dir="./outputs"):
    """
    Run inference on a single image and save the predicted mask.

    Args:
        model (torch.nn.Module): The trained model.
        image_path (str): Path to the input image.
        transform (callable): Transformations for the input image.
        device (torch.device): Device to run the inference on (CPU/GPU).
        output_dir (str): Directory to save the predicted mask.
    """
    model.eval()
    with torch.no_grad():
        # Load and transform the image
        image = Image.open(image_path).convert("L")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Perform inference
        output = model(input_tensor)
        output_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)  # Binarize the output (Only works on cpu)
        
        # Resize the original image to match the predicted mask dimensions
        resized_image = image.resize((output_mask.shape[1], output_mask.shape[0]), Image.LANCZOS)
        
        # Combine resized input image and predicted mask into a single image
        input_array = np.array(resized_image)
        mask_array = output_mask * 255  # Scale mask to 0-255
        combined_image = np.hstack((input_array, mask_array))  # Combine side by side
        
        # Save the combined image
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(image_path)
        combined_path = os.path.join(output_dir, f"combined_{base_name}")
        Image.fromarray(combined_image).save(combined_path)
        print(f"Combined image saved to {combined_path}")
        
        # Optionally visualize the results
        plt.figure(figsize=(12, 6))
        plt.title("Input Image and Predicted Mask")
        plt.imshow(combined_image, cmap="gray")
        plt.axis("off")
        plt.show()
def run_inference_on_folder(model, folder_path, transform, device, output_dir="./outputs"):
    """
    Run inference on all images in a folder and save the predicted masks.

    Args:
        model (torch.nn.Module): The trained model.
        folder_path (str): Path to the folder containing input images.
        transform (callable): Transformations for the input image.
        device (torch.device): Device to run the inference on (CPU/GPU).
        output_dir (str): Directory to save the predicted masks.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all images in the folder
    for image_name in os.listdir(folder_path):
        # Construct full image path
        image_path = os.path.join(folder_path, image_name)

        # Check if the file is an image (optional check for extensions)
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            with torch.no_grad():
                # Load and transform the image
                image = Image.open(image_path).convert("L")
                input_tensor = transform(image).unsqueeze(0).to(device)

                # Perform inference
                output = model(input_tensor)
                output_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)  # Binarize the output (Only works on CPU)

                # Resize the original image to match the predicted mask dimensions
                resized_image = image.resize((output_mask.shape[1], output_mask.shape[0]), Image.LANCZOS)

                # Combine resized input image and predicted mask into a single image
                input_array = np.array(resized_image)
                mask_array = output_mask * 255  # Scale mask to 0-255
                combined_image = np.hstack((input_array, mask_array))  # Combine side by side

                # Save the combined image
                combined_path = os.path.join(output_dir, f"combined_{image_name}")
                Image.fromarray(combined_image).save(combined_path)
                print(f"Combined image saved to {combined_path}")

                # Optionally visualize the results
                # Uncomment the following if you want to see the images as well
                # plt.figure(figsize=(12, 6))
                # plt.title(f"Input Image and Predicted Mask - {image_name}")
                # plt.imshow(combined_image, cmap="gray")
                # plt.axis("off")
                # plt.show()