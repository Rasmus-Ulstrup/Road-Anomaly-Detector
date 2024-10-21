import cv2
import numpy as np
import matplotlib.pyplot as plt
from road_anomaly_detector.Calibration.camera import camProbities
from road_anomaly_detector.Calibration.Image_acqusition import LineScanCamera

def load_image(filepath):
    """Load an image in grayscale mode."""
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Error: Unable to load image from {filepath}")
    return image

def display_image(image, title="Image"):
    """Display an image using Matplotlib."""
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_image(image, save_path):
    """Save an image to the specified path."""
    success = cv2.imwrite(save_path, image)
    if not success:
        raise IOError(f"Error: Unable to save image to {save_path}")
    print(f"Image saved successfully at {save_path}")

def find_paper_contour(image, lower=170, upper=255, blur=True):
    """Detect paper contour in the image using thresholding and morphological operations."""
    if blur:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    thresh = cv2.threshold(image, lower, upper, cv2.THRESH_BINARY)[1]
    
    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image.")

    big_contour = max(contours, key=cv2.contourArea)
    
    return big_contour

def calculate_dimensions_from_contour(image, contour):
    """Approximate a contour to a polygon and calculate its dimensions."""
    peri = cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, 0.04 * peri, True)

    def euclidean_distance(pt1, pt2):
        return np.linalg.norm(np.array(pt1) - np.array(pt2))

    if len(corners) == 4:
        top_left, top_right, bottom_right, bottom_left = [corner[0] for corner in corners]
        height_px = euclidean_distance(top_left, top_right)
        width_px = euclidean_distance(top_left, bottom_left)
    else:
        raise ValueError("The contour does not have exactly 4 corners.")

    polygon = image.copy()
    cv2.polylines(polygon, [corners], True, (0, 0, 255), 2, cv2.LINE_AA)

    return np.round(height_px), np.round(width_px), polygon

def paper_size(image, lower=170, upper=255, blur=True):
    """Find and return paper size (height and width in pixels) and polygon representation."""
    contour = find_paper_contour(image, lower, upper, blur)
    return calculate_dimensions_from_contour(image, contour)

def start_calibration(cam, initial_resolution, image):
    """Execute the paper calibration loop to update the encoder resolution."""
    a4_height_mm, a4_width_mm = 297, 210

    while True:
        # image = load_image("road_anomaly_detector/Calibration/cali1.png")
        height_px, width_px, polygon = paper_size(image)

        height_mm = height_px * cam.getSpartial()
        width_mm = width_px * cam.getSpartial()

        error_height = (abs(height_mm - a4_height_mm) / a4_height_mm) * 100
        error_width = (abs(width_mm - a4_width_mm) / a4_width_mm) * 100

        new_resolution = initial_resolution + (initial_resolution * error_height / 100)
        initial_resolution = new_resolution

        print(f"\nPixels: {height_px} x {width_px}")
        print(f"Real size [mm]: {height_mm:.2f} x {width_mm:.2f}")
        print(f"Error [%]: {error_height:.2f}% x {error_width:.2f}%")
        print(f"Suggested new resolution: {new_resolution:.2f}\n")

        if input("Type 'q' to quit, or any other key to recalibrate: ").lower() == 'q':
            break

def main():
    try:
        print("Starting encoding calibration process...")
        WD = float(input("Enter Work distance (in meters): "))

        cam_specs = {
            "Resolution": (4096, 1),
            "Pixel Size": (3.5, 3.5),
            "Line Rate": 80000,  # Don't matter
            "Wheel Diameter": 0.18
        }

        cam = camProbities(focal=8.5, WD=WD, CamSpecs=cam_specs)
        initial_resolution = cam.calculateEncoderResolution()

        print(f"Initial resolution: {initial_resolution}")
        input("Press Enter to start the paper calibration scheme...")

        camera = LineScanCamera(trigger='', frame_height=1, compression='png')

        # Create instance of the LineScanCamera class with parameters
        #if trigger != encoder it will be software trigger 
        spartial_res_1m = 1 / cam.getSpartial() / 1000 #1 m / spartial_res
        camera = LineScanCamera(trigger='', frame_height=spartial_res_1m, compression='png')

        # Capture and display the image
        image = camera.capture_image()

        start_calibration(cam, initial_resolution, image)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
