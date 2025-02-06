import cv2
import numpy as np

# Load the image
image_path = "./preprocessing/Images/tile1.png" 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Image not found!")
    exit()


# 1. Resize the image to 512x512
image = cv2.resize(image, (512, 512))
cv2.imshow("Original (Resized)", image)

# # Resize the image
# image = cv2.resize(image, (512, 512))

# Callback function for trackbars
def nothing(x):
    pass

cv2.namedWindow("Adjustments")

# CLAHE Parameters
cv2.createTrackbar("CLAHE Clip Limit", "Adjustments", 75, 100, nothing) 
cv2.createTrackbar("CLAHE Tile Size", "Adjustments", 8, 20, nothing) 

# Bilateral filter
cv2.createTrackbar("Bilateral d", "Adjustments", 20, 20, nothing) 
cv2.createTrackbar("Sigma Color", "Adjustments", 31, 300, nothing) 
cv2.createTrackbar("Sigma Space", "Adjustments", 3, 300, nothing)  

# Black Top-Hat
#cv2.createTrackbar("Kernel Size", "Adjustments", 15, 50, nothing)  # Structuring element size

# Canny Edge 
cv2.createTrackbar("Canny Min", "Adjustments", 90, 2000, nothing)
cv2.createTrackbar("Canny Max", "Adjustments", 100, 2000, nothing)

# Threshold
cv2.createTrackbar("Threshold Value", "Adjustments", 30, 255, nothing)  

while True:
    # Get CLAHE 
    clip_limit = cv2.getTrackbarPos("CLAHE Clip Limit", "Adjustments") / 10.0
    tile_size = cv2.getTrackbarPos("CLAHE Tile Size", "Adjustments")
    if tile_size < 1: 
        tile_size = 1

    # Get Bilateral filter parameters
    d = cv2.getTrackbarPos("Bilateral d", "Adjustments")
    if d < 1: 
        d = 1
    sigma_color = cv2.getTrackbarPos("Sigma Color", "Adjustments")
    sigma_space = cv2.getTrackbarPos("Sigma Space", "Adjustments")

    canny_min = cv2.getTrackbarPos("Canny Min", "Adjustments")
    canny_max = cv2.getTrackbarPos("Canny Max", "Adjustments")

    # Get threshold value
    threshold_value = cv2.getTrackbarPos("Threshold Value", "Adjustments")

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    clahe_image = clahe.apply(image)

    # Apply Bilateral Filter
    bilateral_filtered = cv2.bilateralFilter(clahe_image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

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
