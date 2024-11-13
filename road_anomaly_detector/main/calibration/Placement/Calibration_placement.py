# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from road_anomaly_detector.main.camera.Image_acqusition import LineScanCamera
from road_anomaly_detector.main.calibration.Camera_calibration.find_image_coords import find_line_x
import time
import time
import pandas as pd

# Constants
TOLERANCE = 10  # Maximum distance considered to belong to the same line for clustering

# Functions

def PlacementCalibration(image, average_if_even=True):
    # Retrieve x-coordinates (example function, should be defined in your code)
    x_coords = find_line_x(image,threshold_value=120)
    
    # Sort the x-coordinates to ensure distance calculation is meaningful
    x_coords_sorted = sorted(x_coords)
    n = len(x_coords_sorted)
    
    # Check if x_coords_sorted has enough elements
    if n == 0:
        print("Error: x_coords_sorted is empty.")
        return np.array([1,1]), np.array([1,1]), np.array([1])
    elif n == 1:
        middle_value = x_coords_sorted[0]
    elif n % 2 == 1:
        # Odd length, single middle element
        middle_value = x_coords_sorted[n // 2]
    else:
        # Even length, choose based on preference
        if average_if_even:
            # Return the average of the two central elements
            middle_value = (x_coords_sorted[n // 2 - 1] + x_coords_sorted[n // 2]) / 2
        else:
            # Return the two central elements as a tuple
            middle_value = (x_coords_sorted[n // 2 - 1], x_coords_sorted[n // 2])

    # Calculate distances between consecutive x-coordinates
    distances = np.diff(x_coords_sorted) if n > 1 else np.array([])  # Empty if only one element
    
    #print("Sorted x-coordinates:", np.array(x_coords_sorted))
    #print("Distances between consecutive x-coordinates:", distances)

    return x_coords_sorted, distances, middle_value

def update_plot(ax_image, ax_text, ax_distances, image, x_cross, middle_line_coordinates, distances_between_centers):
    """Update the Matplotlib plot with edge detection image, calculated results, and distances."""

    y_positions = np.full_like(x_cross, image.shape[0] // 2)  # Middle row, adjust as needed
    ax_image.clear()
    ax_image.imshow(image, cmap='gray')
    ax_image.set_title("Edge Detection")
    ax_image.set_ylim([1000 - 100, 1100 + 100]) 
    ax_image.scatter(x_cross, y_positions, color='lime', s=20, label="Interpolated Points")  # Overlay points
    ax_image.vlines(x_cross, ymin=0, ymax=image.shape[0] - 1, color='lime', linewidth=1.5, label="Interpolated Lines")
    ax_image.vlines(middle_line_coordinates, ymin=0, ymax=image.shape[0] - 1, color='blue', linewidth=1.5)
    ax_image.axvline(x=1024, color='red', linestyle='--', linewidth=2)

    ax_text.clear()
    ax_text.axis('off')

    # Count the number of distances between centers
    distance_count = len(distances_between_centers)

    text = (f"Middle line x-coordinate:\n{np.array2string(middle_line_coordinates, precision=2, separator=', ')}\n\n"
            f"Distances Between Centers:\n{np.array2string(distances_between_centers, precision=2, separator=', ')}\n\n"
            f"Number of distances between centers: {distance_count}")

    ax_text.text(0.1, 0.5, text, fontsize=10, verticalalignment='center')

    ax_distances.clear()
    ax_distances.plot(distances_between_centers, color='red', marker='o', linestyle='dashed', linewidth=1)
    ax_distances.set_title("Distances Between Centers")
    ax_distances.set_xlabel("Line Number")
    ax_distances.set_ylabel("Distance")
    ax_distances.set_xlim([-1, 61])
    ax_distances.set_ylim([20, 40])
    ax_distances.grid()


def main():
    """Main function to handle the placement calibration loop.."""
    print("Starting dynamic placement calibration, press any key to exit...")

    plt.ion()
    fig, (ax_image, ax_text, ax_distances) = plt.subplots(3, 1, figsize=(10, 8))

    camera = LineScanCamera(trigger='', exposure=40, frame_height=2048, compression='png', gamma=3.98)

    # image = cv2.imread('road_anomaly_detector/main/calibration/Camera_calibration/test_image3.png', cv2.IMREAD_GRAYSCALE)
    # if image is not None:
    #     print("Image loaded successfully")
    # else:
    #     print("Image loading failed")

    while True:
        image = camera.capture_image()
        

        # Perform placement calibration
        x_coords, distances, middle = PlacementCalibration(image)
        

        # Update the plot with new data
        update_plot(ax_image, ax_text, ax_distances, image, x_coords, middle, distances)

        plt.draw()
        time.sleep(0.1)

        if plt.waitforbuttonpress(timeout=0.1):
            print("Exiting placement calibration loop")
            camera.cleanup()
            break

    # Save data to CSV files and image after exiting the loop
    if x_coords is not None and distances is not None and middle is not None:
        # Save x_coords to CSV
        pd.DataFrame({"x_coords": x_coords}).to_csv("x_coords.csv", index=False)
        print("x_coords saved to x_coords.csv")

        # Save distances to CSV
        pd.DataFrame({"distances": distances}).to_csv("distances.csv", index=False)
        print("distances saved to distances.csv")

        # Save middle to CSV (if middle is a list, else wrap it in a list)
        middle_data = middle if isinstance(middle, (list, np.ndarray)) else [middle]
        pd.DataFrame({"middle": middle_data}).to_csv("middle.csv", index=False)
        print("middle saved to middle.csv")

        # Save the last captured image
        cv2.imwrite("calibration_image.png", image)
        print("Image saved as calibration_image.png")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
