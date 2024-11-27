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
def cross_ratio(unit):
    if unit.size != 4 or unit.ndim != 1:
        raise ValueError("Input must be a 1D numpy array with exactly four elements.")
    
    A, B, C, D = unit
    return ((C-A) / (C-B)) / ((D-A) / (D-B))

def feature_points_correspondences(X_image_coordinates, h, l):
    # P D P D P D P

    # Pad last vertical line based on ca = da * 2
    ca_last = X_image_coordinates[-1] - X_image_coordinates[-3]
    X_image_coordinates = np.append(X_image_coordinates, [0, X_image_coordinates[-1] + ca_last])
    
    #Find all diagonal X in pattern coordinates
    i = 1
    diagonal = []
    for i_cross in range(0, len(X_image_coordinates) - 4, 2):  # Move 2 steps forward each iteration
        # Find ABCD according to figure 2 in paper
        ABCD = X_image_coordinates[[i_cross, i_cross+1, i_cross+2, i_cross+4]] # A B C D
        # Cross ratio
        r_abcd = cross_ratio(ABCD)
        eta = (2*r_abcd-2) / (2*r_abcd-1)
        # Find X
        x_b = eta * h + (i-1)*h
        # Find Y
        y_b = (l/h) * x_b - (i-1) * l
        # Next diagonal line
        i = i + 1
        #append
        diagonal.append([x_b, y_b])

    #Find all other lines
    #print(diagonal)
    diagonal = np.array(diagonal) 
    vertical = []
    for i in range(0, len(diagonal), 2):
        x_a = i*h
        if i != 0:
            x_c = (i+1)*h
        else:
            x_c = 1*h

        # Extract coordinates from diagonals
        if i != len(diagonal)-1:
            dia_points = diagonal[[i,i+1]]
        else:
            dia_points = diagonal[[i-1, i]]

        x1, y1 = dia_points[0]
        x2, y2 = dia_points[1]
        # Calculate slope (m)
        m = (y2 - y1) / (x2 - x1)
        # Calculate y-intercept (b)
        b = y1 - m * x1
        # Calculate y
        y_a = m * x_a + b
        y_c = m * x_c + b
        # append
        vertical.append([x_a,y_a])
        vertical.append([x_c, y_c])

    vertical = np.array(vertical)
    # Create an empty array for interlaced coordinates
    pattern_coordinates = np.empty((diagonal.shape[0] + vertical.shape[0], 2), dtype=diagonal.dtype)

    # Interlace arrays
    pattern_coordinates[0::2] = vertical  # Place `vertical` elements at even rows
    pattern_coordinates[1::2] = diagonal  # Place `diagonal` elements at odd rows
    #print(diagonal.shape)
    #print(vertical.shape)
    return pattern_coordinates

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
    h = 0.025
    l = 0.200

    pattern_space = feature_points_correspondences(np.array(x_coords),h,l)

    return x_coords_sorted, distances, middle_value, pattern_space

def update_plot(ax_image, ax_pattern, ax_distances, image, x_cross, middle_line_coordinates, distances_between_centers, pattern):
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
    #print(pattern)
    ax_pattern.clear()
    ax_pattern.plot(pattern[:, 0],pattern[:, 1], color='red', marker='o', linestyle='dashed', linewidth=1)
    ax_pattern.set_title("Distances Between Centers")
    ax_pattern.set_xlabel("Line Number")
    ax_pattern.set_ylabel("Distance")
    #ax_pattern.set_xlim([-1, 121])
    ax_pattern.set_ylim([-0, 0.2])
    ax_pattern.grid()


    ax_distances.clear()
    ax_distances.plot(distances_between_centers, color='red', marker='o', linestyle='dashed', linewidth=1)
    ax_distances.set_title("Distances Between Centers")
    ax_distances.set_xlabel("Line Number")
    ax_distances.set_ylabel("Distance")
    ax_distances.set_xlim([-1, 121])
    ax_distances.set_ylim([10, 20])
    ax_distances.grid()


def main():
    """Main function to handle the placement calibration loop.."""
    print("Starting dynamic placement calibration, press any key to exit...")

    plt.ion()
    fig, (ax_image, ax_text, ax_distances) = plt.subplots(3, 1, figsize=(10, 8))

    #camera = LineScanCamera(trigger='', exposure=40, frame_height=2048, compression='png', gamma=3.98)

    image = cv2.imread('road_anomaly_detector/main/calibration/Camera_calibration/calibration_image_0.png', cv2.IMREAD_GRAYSCALE)
    if image is not None:
        print("Image loaded successfully")
    else:
        print("Image loading failed")

    while True:
        #image = camera.capture_image()
        

        # Perform placement calibration
        x_coords, distances, middle, pattern = PlacementCalibration(image)
        

        # Update the plot with new data
        update_plot(ax_image, ax_text, ax_distances, image, x_coords, middle, distances, pattern)

        plt.draw()
        time.sleep(0.1)

        if plt.waitforbuttonpress(timeout=0.1):
            print("Exiting placement calibration loop")
            #camera.cleanup()
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