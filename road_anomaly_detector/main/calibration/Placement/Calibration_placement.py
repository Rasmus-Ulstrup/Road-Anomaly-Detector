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
        # Return an empty array if there are no coordinates
        return np.array([])
    
    unique_x_coords = np.unique(coordinates)
    
    if unique_x_coords.size == 0:
        # If no unique coordinates are found, return an empty array
        return np.array([])

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
        # If no line centers are found, return an empty array or some default value
        print("Warning: No line centers detected.")
        return np.array([])

    line_count = (coordinates.size -1 )/ 2

    if isinstance(line_count, float) and line_count > 1:
        middle_coords = np.ones(2)
        middle_coords[0] = coordinates[int(np.floor(line_count))]
        middle_coords[1] = coordinates[int(np.ceil(line_count))]
    else:
        middle_coords = coordinates[int(line_count)]

    return middle_coords

def PlacementCalibration(image):
    """Perform placement calibration using edge detection to find line centers and distances."""
    edges = cv2.Canny(image, 50, 100)  # Perform edge detection
    coordinates = np.where(edges >= 250)[1]  # Get x-coordinates of edge points

    # Find line centers and calculate distances between them
    line_centers = find_line_centers(coordinates)
    distances_between_centers = np.diff(line_centers)

    # Calculate middle line coordinates
    middle_line_coordinates = calculate_middle_line(line_centers)

    return middle_line_coordinates, distances_between_centers, edges

def update_plot(ax_image, ax_text, ax_distances, edges, middle_line_coordinates, distances_between_centers):
    """Update the Matplotlib plot with edge detection image, calculated results, and distances."""
    ax_image.clear()
    ax_image.imshow(edges, cmap='gray')
    ax_image.set_title("Edge Detection")
    ax_image.axvline(x=1024, color='red', linestyle='--', linewidth=2)

    # Prepare and display text content
    ax_text.clear()
    ax_text.axis('off')

    # Count the number of distances between centers
    distance_count = len(distances_between_centers)

    text = (f"Middle line x-coordinate:\n{np.array2string(middle_line_coordinates, precision=2, separator=', ')}\n\n"
            f"Distances Between Centers:\n{np.array2string(distances_between_centers, precision=2, separator=', ')}\n\n"
            f"Number of distances between centers: {distance_count}")

    ax_text.text(0.1, 0.5, text, fontsize=10, verticalalignment='center')

    # Plot the distances between centers
    ax_distances.clear()
    ax_distances.plot(distances_between_centers, marker='o')
    ax_distances.set_title("Distances Between Centers")
    ax_distances.set_xlabel("Line Number")
    ax_distances.set_ylabel("Distance")
    ax_distances.set_xlim([-1, 42])
    ax_distances.set_ylim([20, 40])


def main():
    """Main function to handle the placement calibration loop."""
    print("Starting dynamic placement calibration, press any key to exit...")

    # Set up interactive plot with three subplots
    plt.ion()
    fig, (ax_image, ax_text, ax_distances) = plt.subplots(3, 1, figsize=(10, 8))

    # Create instance of the LineScanCamera class with parameters
    #if trigger != encoder it will be software triggerq
    camera = LineScanCamera(trigger='', exposure=50, frame_height=100, compression='png')

    while True:
        # Create a synthetic test image with black lines
        #height, width = 100, 2048
        #image = np.ones((height, width))
        #for x in range(0, width, 20):
        #    image[:, x] = 0  # Draw black vertical lines
        #image = (image * 255).astype(np.uint8)  # Convert to 8-bit grayscale

        # Capture image
        image = camera.capture_image()

        # Display image
        # camera.show_image()

        # Perform placement calibration
        middle_line_coordinates, distances_between_centers, edges = PlacementCalibration(image)

        # Update the plot with new data
        update_plot(ax_image, ax_text, ax_distances, image, middle_line_coordinates, distances_between_centers)

        # Draw the updated plot and pause to allow display to refresh
        plt.draw()

        # Sleep
        time.sleep(0.05)  # 100 ms refresh rate (10 Hz)

        # Exit the loop if a button is pressedq
        if plt.waitforbuttonpress(timeout=0.1):
            print("Exiting placement calibration loop")
            camera.cleanup()
            break

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Display the final plot

if __name__ == "__main__":
    main()
