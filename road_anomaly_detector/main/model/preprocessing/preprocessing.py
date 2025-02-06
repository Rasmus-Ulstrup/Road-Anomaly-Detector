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

# Gabor filters
ksize = 31 
sigma = 2.0 
lambd = 10.0 
gamma = 1 
psi = 0 

# multiple orientations
orientations = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3]

# Gabor filters for each orientation
gabor_sum = np.zeros_like(image, dtype=np.float32)  # Initialize the sum

for theta in orientations:
    # Create Gabor kernel
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    
    # Filter image using the Gabor kernel
    filtered = cv2.filter2D(image, cv2.CV_32F, gabor_kernel)
    
    # Accumulate the absolute response
    gabor_sum += np.abs(filtered)

# Normalize the summed responses to [0, 255] for visualization
gabor_sum_normalized = cv2.normalize(gabor_sum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# Display the result
cv2.imshow("Gabor Filtered (Summed Responses)", gabor_sum_normalized)


# Global Histogram
# equalized = cv2.equalizeHist(image)
# cv2.imshow("Global Histogram Equalization", equalized)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=75.0, tileGridSize=(8,8))
clahe_image = clahe.apply(image)
cv2.imshow("CLAHE Image", clahe_image)

# Bilateral filter to preserve edges while smoothing
bilateral = cv2.bilateralFilter(clahe_image, d=20, sigmaColor=31, sigmaSpace=5)
cv2.imshow("Bilateral Filtered", bilateral)


cv2.waitKey(0)
cv2.destroyAllWindows()






# image = cv2.GaussianBlur(image, (5, 5), 0)

# # Display the original image
# cv2.imshow("Original Image", image)

# # Histogram Equalization
# equalized_image = cv2.equalizeHist(image)
# cv2.imshow("Histogram Equalization", equalized_image)

# # CLAHE (Contrast Limited Adaptive Histogram Equalization)
# clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
# clahe_image = clahe.apply(image)
# cv2.imshow("CLAHE", clahe_image)

# # Adaptive Contrast Enhancement
# normalized_image = cv2.normalize(image, None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX)
# cv2.imshow("Adaptive Contrast Enhancement", normalized_image)

# # Unsharp Masking
# blurred = cv2.GaussianBlur(clahe_image, (9, 9), 10)
# sharpness = 1
# unsharp_image = cv2.addWeighted(clahe_image, 1 + sharpness, blurred, -0.5, 0)
# cv2.imshow("Unsharp Masking", unsharp_image)

# # Apply Image Smoothing: Bilateral Filter
# bilateral_filtered_image = cv2.bilateralFilter(unsharp_image, d=9, sigmaColor=35, sigmaSpace=16)
# cv2.imshow("Bilateral Filter", bilateral_filtered_image)

# # Apply Otsu's Thresholding
# def apply_otsu_threshold(img, window_name):
#     _, otsu_image = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
#     cv2.imshow(window_name + " - Otsu Threshold", otsu_image)

# #apply_otsu_threshold(equalized_image, "Histogram Equalization")
# #apply_otsu_threshold(clahe_image, "CLAHE")
# #apply_otsu_threshold(normalized_image, "Adaptive Contrast Enhancement")
# apply_otsu_threshold(bilateral_filtered_image, "Bilateral Filter")



# # # Wait for a key press and close all windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()
