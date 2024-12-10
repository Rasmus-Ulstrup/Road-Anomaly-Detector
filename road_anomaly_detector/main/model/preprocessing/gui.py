import cv2
import numpy as np

# Load the image
image_path = "./preprocessing/Images/tile1.png"  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Image not found!")
    exit()


# 1. Resize the image to 512x512
image = cv2.resize(image, (512, 512))
cv2.imshow("Original (Resized)", image)

# Resize the image
image = cv2.resize(image, (512, 512))

# Callback function for trackbars (required by OpenCV)
def nothing(x):
    pass

# Create a window
cv2.namedWindow("Adjustments")

# CLAHE Parameters
cv2.createTrackbar("CLAHE Clip Limit", "Adjustments", 75, 100, nothing)  # Scale 0.1-10 (multiply by 0.1)
cv2.createTrackbar("CLAHE Tile Size", "Adjustments", 8, 20, nothing)  # Range 1-20

# Bilateral Filter Parameters
cv2.createTrackbar("Bilateral d", "Adjustments", 20, 20, nothing)  # Neighborhood size
cv2.createTrackbar("Sigma Color", "Adjustments", 31, 300, nothing)  # Range 0-300
cv2.createTrackbar("Sigma Space", "Adjustments", 3, 300, nothing)  # Range 0-300

# Black Top-Hat Parameters
#cv2.createTrackbar("Kernel Size", "Adjustments", 15, 50, nothing)  # Structuring element size

# Canny Edge Parameters
cv2.createTrackbar("Canny Min", "Adjustments", 90, 2000, nothing)  # Minimum threshold
cv2.createTrackbar("Canny Max", "Adjustments", 100, 2000, nothing)  # Maximum threshold

# Threshold Parameters
cv2.createTrackbar("Threshold Value", "Adjustments", 30, 255, nothing)  # Binarization threshold

while True:
    # Get CLAHE parameters from the trackbars
    clip_limit = cv2.getTrackbarPos("CLAHE Clip Limit", "Adjustments") / 10.0
    tile_size = cv2.getTrackbarPos("CLAHE Tile Size", "Adjustments")
    if tile_size < 1:  # Avoid invalid tile size
        tile_size = 1

    # Get Bilateral filter parameters from the trackbars
    d = cv2.getTrackbarPos("Bilateral d", "Adjustments")
    if d < 1:  # Avoid invalid d
        d = 1
    sigma_color = cv2.getTrackbarPos("Sigma Color", "Adjustments")
    sigma_space = cv2.getTrackbarPos("Sigma Space", "Adjustments")

    # Get Black Top-Hat kernel size
    # kernel_size = cv2.getTrackbarPos("Kernel Size", "Adjustments")
    # if kernel_size < 1:  # Ensure kernel size is valid
    #     kernel_size = 1
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Get Canny edge detection thresholds
    canny_min = cv2.getTrackbarPos("Canny Min", "Adjustments")
    canny_max = cv2.getTrackbarPos("Canny Max", "Adjustments")

    # Get threshold value
    threshold_value = cv2.getTrackbarPos("Threshold Value", "Adjustments")

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    clahe_image = clahe.apply(image)

    # Apply Bilateral Filter
    bilateral_filtered = cv2.bilateralFilter(clahe_image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Apply Black Top-Hat Filter
    #black_tophat = cv2.morphologyEx(clahe_image, cv2.MORPH_BLACKHAT, kernel)

    # Apply Thresholding
    _, threshold = cv2.threshold(bilateral_filtered, threshold_value, 255, cv2.THRESH_BINARY)

    # Apply Canny Edge Detection
    edges = cv2.Canny(threshold, canny_min, canny_max)
    

    # Display results
    cv2.imshow("Original Image", image)
    cv2.imshow("CLAHE Image", clahe_image)
    cv2.imshow("Bilateral Filtered", bilateral_filtered)
    #cv2.imshow("Black Top-Hat", black_tophat)
    cv2.imshow("Canny Edges", edges)
    cv2.imshow("Threshold", threshold)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
