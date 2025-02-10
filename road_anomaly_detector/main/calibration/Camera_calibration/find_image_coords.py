import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def find_line_x(image,threshold_value=100):
    _, binary_image = cv2.threshold(image, threshold_value, 1, cv2.THRESH_BINARY_INV)

    mean_binary_image = binary_image.mean(axis=0).astype(np.uint8)  # Convert to uint8 if needed
    binary_row = mean_binary_image 
    #print(binary_row)

    # List to store start and end of clusters
    clusters = []

    # Find clusters of 1's in the row
    start = None  # Start of a cluster
    for x in range(len(binary_row)):
        if binary_row[x] == 1 and (x == 0 or binary_row[x - 1] == 0):  # Start of a cluster
            start = x
        elif binary_row[x] == 0 and start is not None:  # End of a cluster
            clusters.append((start, x - 1))
            start = None

    mean_image = image.mean(axis=0).astype(np.uint8)  # Convert to uint8 if needed
    k = 10
    x_cross = []
    for start_x, end_x in clusters:
        #print(f"Cluster from x = {start_x} to x = {end_x}")
        y = mean_image[start_x - k : end_x + k]
        x = range(0, len(y))
        if len(y) > 5:
            tck = interpolate.splrep(x, y, s=0, k=2) 

            x_new = np.linspace(min(x), max(x), 100)
            y_fit = interpolate.BSpline(*tck)(x_new)
            

            min_value = np.min(y_fit)
            min_index = np.argmin(y_fit)
            x_at_min = x_new[min_index]

            #print(f"The minimum value of y_fit is: {min_value}")
            #print(f"This minimum value occurs at x = {x_at_min}")

            x_cross.append(x_at_min + start_x - k)
        else:
            x_cross.append(0)

        plt.plot(x, y, 'ro', label="original")
        plt.plot(x, y, 'b', label="linear interpolation")
        plt.title("Target data")
        plt.legend(loc='best', fancybox=True, shadow=True)
        plt.grid()
        plt.show() 

        plt.title("BSpline curve fitting")
        plt.plot(x, y, 'ro', label="original")
        plt.plot(x_new, y_fit, '-c', label="B-spline")
        plt.legend(loc='best', fancybox=True, shadow=True)
        plt.grid()
        plt.show()


    return x_cross

def plot_cross_dot(image, x_cross):
    # Set y-coordinates for overlaying, e.g., choose a specific row or area
    y_positions = np.full_like(x_cross, image.shape[0] // 2)  # Middle row, adjust as needed

    # Set the y-axis limit for visualization
    y_limit = (0, 50)

    # Display the images side by side
    plt.figure(figsize=(10, 5))

    # Show the image with float x values as overlayed dots
    plt.imshow(image, cmap='gray')
    plt.scatter(x_cross, y_positions, color='lime', s=20, label="Interpolated Points")  # Overlay points
    plt.vlines(x_cross, ymin=0, ymax=image.shape[0] - 1, color='lime', linewidth=1.5, label="Interpolated Lines")
    plt.title('Image with Overlayed Interpolated Points')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    #image = cv2.imread('road_anomaly_detector\main\calibration\Camera_calibration\calibration_image_0.png', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('road_anomaly_detector\main\calibration\Camera_calibration\calibration_image.png', cv2.IMREAD_GRAYSCALE)

    x_cross = find_line_x(image,160)
    y_positions = np.full_like(x_cross, image.shape[0] // 2)

    #plot_cross_dot(image, x_cross)
    n = len(x_cross)
    distances = np.diff(x_cross) if n > 1 else np.array([])
    print(distances)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # First subplot
    ax_image = axes[0]
    ax_image.imshow(image, cmap='gray')
    ax_image.scatter(x_cross, y_positions, color='lime', s=20, label="Interpolated Points")  # Overlay points
    ax_image.vlines(x_cross, ymin=0, ymax=image.shape[0] - 1, color='lime', linewidth=1.5, label="Interpolated Lines")
    ax_image.set_title('Image with Overlayed Interpolated Points')
    ax_image.axis('off')

    # Second subplot
    ax_distances = axes[1]
    ax_distances.plot(distances, color='red', marker='o', linestyle='dashed', linewidth=1, label="Distances")
    ax_distances.set_title("Distances Plot")
    ax_distances.set_xlabel("Index")
    ax_distances.set_ylabel("Distance")
    ax_distances.set_xlim([-1, len(distances)])
    ax_distances.set_ylim([min(distances) - 1, max(distances) + 1])
    ax_distances.grid()
    ax_distances.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()



