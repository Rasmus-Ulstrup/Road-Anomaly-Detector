# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from road_anomaly_detector.main.camera.Image_acqusition import LineScanCamera
import time

# Constants
TOLERANCE = 10  # Maximum distance considered to belong to the same line for clustering

# Functions
def save_image(image, save_path):
    """Save an image to the specified path."""
    success = cv2.imwrite(save_path, image)
    if not success:
        raise IOError(f"Error: Unable to save image to {save_path}")
    print(f"Image saved successfully at {save_path}")


def find_line_centers(coordinates, tolerance=TOLERANCE):
    """Clusters coordinates by proximity and returns the centers of the lines."""
    if coordinates.size == 0:
        return np.array([])  # Return an empty array if there are no coordinates
    
    unique_x_coords = np.unique(coordinates)
    
    if unique_x_coords.size == 0:
        return np.array([])  # Return an empty array if no unique coordinates are found

    line_centers = []
    current_cluster = [unique_x_coords[0]]
    
    for coord in unique_x_coords[1:]:
        if coord - current_cluster[-1] <= tolerance:
            current_cluster.append(coord)
        else:
            line_centers.append(np.mean(current_cluster))
            current_cluster = [coord]

    # Append the last cluster's center
    if current_cluster:
        line_centers.append(np.mean(current_cluster))

    return np.array(line_centers)


def calculate_middle_line(coordinates):
    """Calculate the coordinates of the middle line based on line centers."""
    if coordinates.size == 0:
        print("Warning: No line centers detected.")
        return np.array([])  # Return an empty array or some default value if no line centers are found

    line_count = (coordinates.size - 1) / 2

    if isinstance(line_count, float) and line_count > 1:
        middle_coords = np.array([coordinates[int(np.floor(line_count))], coordinates[int(np.ceil(line_count))]])
    else:
        middle_coords = np.array([coordinates[int(line_count)]])

    return middle_coords


def PlacementCalibration(image):
    """Perform placement calibration using Canny and Hough Transform with subpixel interpolation."""
    
    # Step 1: Edge detection using Canny
    low_threshold = 150
    high_threshold = 180
    edges = cv2.Canny(image, low_threshold, high_threshold)
    
    # Step 2: Hough Line Transform on Canny edges
    threshold = 50
    min_line_length = 50
    max_line_gap = 20
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    # Step 2.5: Visualize detected lines
    line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Check if there are lines detected
    if lines is not None:
        # Determine the middle line's index in the sorted list of x-coordinates
        x_coords = np.array([(line[0][0] + line[0][2]) // 2 for line in lines])  # Average x-coordinates of lines
        sorted_indices = np.argsort(x_coords)
        middle_index = sorted_indices[len(sorted_indices) // 2]  # Middle index in the sorted list

        # Draw each line, coloring the middle line in red and others in green
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            
            # Draw the middle line in red and the rest in green
            if i == middle_index:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for middle line
            else:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for other lines

    
    # Step 3: Detect line centers with subpixel interpolation
    subpixel_centers = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if the line is vertical (or close to vertical)
            if abs(x2 - x1) < 5:
                # Extract a region around the line in the x-direction
                region = image[min(y1, y2):max(y1, y2), max(0, x1 - 3):min(image.shape[1], x1 + 3)]
                
                # Compute the intensity profile in the x-direction by summing over y
                x_profile = np.sum(region, axis=0)
                
                # Find the subpixel center in the x-direction using a weighted average
                x_indices = np.arange(len(x_profile))
                weighted_sum = np.sum(x_indices * x_profile)
                sum_profile = np.sum(x_profile)
                
                if sum_profile > 0:
                    subpixel_x = x1 + (weighted_sum / sum_profile - (len(x_profile) / 2))
                    subpixel_centers.append(subpixel_x)

    # Cluster subpixel centers to find distinct line centers
    line_centers = find_line_centers(np.array(subpixel_centers))
    distances_between_centers = np.diff(line_centers)

    # Calculate middle line coordinates
    middle_line_coordinates = calculate_middle_line(line_centers)

    return middle_line_coordinates, distances_between_centers, edges, line_image


def update_plot(ax_image, ax_text, ax_distances, edges, middle_line_coordinates, distances_between_centers):
    """Update the Matplotlib plot with edge detection image, calculated results, and distances."""
    ax_image.clear()
    ax_image.imshow(edges, cmap='gray')
    ax_image.set_title("Edge Detection")
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
    ax_distances.set_xlim([-1, 42])
    ax_distances.set_ylim([20, 40])


def main():
    """Main function to handle the placement calibration loop."""
    print("Starting dynamic placement calibration, press any key to exit...")

    plt.ion()
    fig, (ax_image, ax_text, ax_distances) = plt.subplots(3, 1, figsize=(10, 8))

    camera = LineScanCamera(trigger='', exposure=25, frame_height=100, compression='png')

    #image = cv2.imread('road_anomaly_detector/main/calibration/Placement/testimage.png', cv2.IMREAD_GRAYSCALE)
    # if image is not None:
    #     print("Image loaded successfully")
    # else:
    #     print("Image loading failed")

    while True:
        image = camera.capture_image()
        

        # Perform placement calibration
        middle_line_coordinates, distances_between_centers, edges, lineImage = PlacementCalibration(image)

        # Update the plot with new data
        update_plot(ax_image, ax_text, ax_distances, lineImage, middle_line_coordinates, distances_between_centers)

        plt.draw()
        time.sleep(0.05)

        if plt.waitforbuttonpress(timeout=0.1):
            print("Exiting placement calibration loop")
            camera.cleanup()
            break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
