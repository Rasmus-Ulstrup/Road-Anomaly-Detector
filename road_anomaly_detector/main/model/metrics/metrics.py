import numpy as np
import torch
from sklearn.metrics import f1_score
from scipy.spatial.distance import directed_hausdorff

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

def run_inference(model, image_path, transform, device):
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert("L")
        input_tensor = transform(image).unsqueeze(0).to(device)
        output = model(input_tensor)
        output = model(input_tensor).squeeze().cpu().numpy()
        return output
    