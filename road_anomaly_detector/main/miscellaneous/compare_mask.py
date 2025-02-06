import argparse
import cv2
import numpy as np
import sys
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare two semantic masks and visualize TP, FP, FN, TN with percentages and a legend.')
    parser.add_argument('predicted_mask', type=str, help='Path to the predicted mask image.')
    parser.add_argument('reference_mask', type=str, help='Path to the reference (ground truth) mask image.')
    parser.add_argument('--output', type=str, default='comparison_result.png', help='Path to save the comparison result image.')
    return parser.parse_args()

def load_mask(path):
    if not os.path.exists(path):
        print(f"Error: File '{path}' does not exist.")
        sys.exit(1)
    # Load as grayscale
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Unable to read image '{path}'. Ensure it's a valid image file.")
        sys.exit(1)
    # Binarize the mask (assuming threshold 127)
    _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    return binary_mask

def compare_masks(pred, ref):
    # Ensure both masks have the same shape
    if pred.shape != ref.shape:
        print("Error: Predicted mask and reference mask have different dimensions.")
        sys.exit(1)
    
    # Calculate TP, FP, FN, TN
    TP = np.logical_and(pred == 1, ref == 1)
    FP = np.logical_and(pred == 1, ref == 0)
    FN = np.logical_and(pred == 0, ref == 1)
    TN = np.logical_and(pred == 0, ref == 0)
    
    return TP, FP, FN, TN

def calculate_percentages(TP, FP, FN, TN):
    total = TP.size
    tp_percent = (np.sum(TP) / total) * 100
    fp_percent = (np.sum(FP) / total) * 100
    fn_percent = (np.sum(FN) / total) * 100
    tn_percent = (np.sum(TN) / total) * 100
    return tp_percent, fp_percent, fn_percent, tn_percent

def add_legend(image, colors, labels, box_size=50, spacing=100, margin=100):
    """
    Adds a legend to the image.

    Parameters:
    - image: The image to draw the legend on.
    - colors: List of BGR color tuples.
    - labels: List of labels corresponding to the colors.
    - box_size: Size of the color boxes.
    - spacing: Space between boxes and labels.
    - margin: Margin from the edges of the image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    font_color = (255, 255, 255)  # White
    font_thickness = 5
    line_type = cv2.LINE_AA

    # Starting position
    x_start = margin
    y_start = image.shape[0] - margin - (box_size + spacing) * len(labels)

    for i, (color, label) in enumerate(zip(colors, labels)):
        y_position = y_start + i * (box_size + spacing)
        
        # Draw the colored box
        cv2.rectangle(image, 
                      (x_start, y_position), 
                      (x_start + box_size, y_position + box_size), 
                      color, 
                      cv2.FILLED)
        
        # Put the label next to the box
        cv2.putText(image, label, 
                    (x_start + box_size + 10, y_position + box_size - 5), 
                    font, 
                    font_scale, 
                    font_color, 
                    font_thickness, 
                    line_type)

def create_visualization(TP, FP, FN, TN, percentages, ref_mask):
    tp_percent, fp_percent, fn_percent, tn_percent = percentages
    # Create a blank color image
    height, width = TP.shape
    visualization = np.zeros((height, width, 3), dtype=np.uint8)

    # Define colors in BGR
    COLOR_TP = (0, 255, 0)      # Green
    COLOR_FP = (0, 0, 255)      # Red
    COLOR_FN = (0, 255, 255)    # Yellow
    COLOR_TN = (0, 0, 0)        # Black

    # Assign colors
    visualization[TP] = COLOR_TP
    visualization[FP] = COLOR_FP
    visualization[FN] = COLOR_FN
    visualization[TN] = COLOR_TN  # Optional: You can make TN transparent if desired

    # Convert reference mask to BGR for blending
    ref_bgr = cv2.cvtColor(ref_mask * 255, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(ref_bgr, 0.5, visualization, 0.5, 0)

    # Prepare text to overlay
    text_lines = [
        f"True Positives (TP): {tp_percent:.2f}%",
        f"False Positives (FP): {fp_percent:.2f}%",
        f"False Negatives (FN): {fn_percent:.2f}%",
        f"True Negatives (TN): {tn_percent:.2f}%"
    ]

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    font_color = (255, 255, 255)  # White
    font_thickness = 2
    line_type = cv2.LINE_AA

    # Calculate position for the text (right corner with some padding)
    padding = 10
    line_height = 100  # Approximate height per line
    for i, line in enumerate(text_lines):
        text_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        text_x = width - text_size[0] - padding
        text_y = padding + (i + 1) * line_height
        # Add a black background for better visibility
        cv2.rectangle(blended, 
                      (text_x - 5, text_y - text_size[1] - 5), 
                      (text_x + text_size[0] + 5, text_y + 5), 
                      (0, 0, 0), 
                      cv2.FILLED)
        # Put the text on top
        cv2.putText(blended, line, 
                    (text_x, text_y), 
                    font, 
                    font_scale, 
                    font_color, 
                    font_thickness, 
                    line_type)
    
    # Add Legend to the bottom left corner
    legend_colors = [COLOR_TP, COLOR_FP, COLOR_FN, COLOR_TN]
    legend_labels = ['True Positives (TP)', 'False Positives (FP)', 
                    'False Negatives (FN)', 'True Negatives (TN)']
    add_legend(blended, legend_colors, legend_labels, box_size=80, spacing=10, margin=1024)

    return blended

def main():
    args = parse_arguments()
    
    # Load masks
    pred_mask = load_mask(args.predicted_mask)
    ref_mask = load_mask(args.reference_mask)
    
    # Compare masks
    TP, FP, FN, TN = compare_masks(pred_mask, ref_mask)
    
    # Calculate percentages
    tp_percent, fp_percent, fn_percent, tn_percent = calculate_percentages(TP, FP, FN, TN)
    
    # Create visualization with percentages and legend
    visualization = create_visualization(TP, FP, FN, TN, 
                                         (tp_percent, fp_percent, fn_percent, tn_percent), 
                                         ref_mask)
    
    # Save the result
    cv2.imwrite(args.output, visualization)
    print(f"Comparison result saved to '{args.output}'.")
    
    # Optionally, display the image
    # cv2.imshow('Comparison Result', visualization)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
