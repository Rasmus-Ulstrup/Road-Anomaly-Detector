import cv2
import numpy as np
import matplotlib.pyplot as plt
from road_anomaly_detector.main.calibration.camera import camProbities
from road_anomaly_detector.main.camera.Image_acqusition import LineScanCamera

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

def save_image(image, save_path):
    """Save an image to the specified path."""
    success = cv2.imwrite(save_path, image)
    if not success:
        raise IOError(f"Error: Unable to save image to {save_path}")
    print(f"Image saved successfully at {save_path}")

def find_paper_contour(image, lower=170, upper=255, blurEn=True, morphEn=True):
    """Detect paper contour in the image using thresholding and morphological operations."""
    if blurEn:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    thresh = cv2.threshold(image, lower, upper, cv2.THRESH_BINARY)[1]
    
    if morphEn:
        kernel = np.ones((7, 7), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    else:
        morph = thresh

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image.")

    big_contour = max(contours, key=cv2.contourArea)
    
    return big_contour, thresh, morph, contours

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
        if width_px > height_px:
            height_px, width_px = width_px, height_px
    else:
        raise ValueError("The contour does not have exactly 4 corners.")

    polygon = image.copy()
    cv2.polylines(polygon, [corners], True, (0, 0, 255), 2, cv2.LINE_AA)

    return np.round(height_px), np.round(width_px), polygon

def display_stages(image, thresh, morph, contours, polygon):
    """Display the thresh, morph, contours, and polygon stages in one figure."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Display thresholded image
    axes[0, 0].imshow(thresh, cmap='gray')
    axes[0, 0].set_title("Threshold")
    axes[0, 0].axis('off')

    # Display morphology result
    axes[0, 1].imshow(morph, cmap='gray')
    axes[0, 1].set_title("Morphology")
    axes[0, 1].axis('off')

    # Display contours
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    axes[1, 0].imshow(contour_image, cmap='gray')
    axes[1, 0].set_title("Contours")
    axes[1, 0].axis('off')

    # Display polygon
    axes[1, 1].imshow(polygon, cmap='gray')
    axes[1, 1].set_title("Polygon")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

def paper_size(image, lower=170, upper=255, dispay=True, blurEn=True, morphEn=True):
    """Find and return paper size (height and width in pixels) and polygon representation."""
    contour, thresh, morph, contours = find_paper_contour(image, lower, upper, blurEn, morphEn)
    height_px, width_px, polygon = calculate_dimensions_from_contour(image, contour)
    if dispay:
        display_stages(image, thresh, morph, contours, polygon)
    return height_px, width_px, polygon

def main():
    try:
        print("Starting encoding calibration process...")
        #WD = float(input("Enter Work distance (in meters): "))
        WD = 0.764

        cam_specs = {
            "Resolution": (4096/2, 1),
            "Pixel Size": (3.5*2, 3.5*2),
            "Line Rate": 80000,  # Don't matter
            "Wheel Diameter": 0.18
        }

        cam = camProbities(focal=8.5, WD=WD, CamSpecs=cam_specs)
        initial_resolution = cam.calculateEncoderResolution()

        print(f"Initial resolution: {initial_resolution}")
        input("Press Enter to start the paper calibration scheme...")

        spartial_res_1m = 1 / (cam.getSpartial() / 1000) # 1 m / spartial_res
        #print(cam.getSpartial() / 1000)
        #camera = LineScanCamera(trigger='encoder', exposure=10, frame_height=spartial_res_1m, compression='png')

        a3_height_mm, a3_width_mm = 420*2, 297

        while True:
            image = load_image("road_anomaly_detector/main/calibration/enoder_test_1.png")
            #image = camera.capture_image()

            # Display image
            print("picture taken...")
            #display_image(image)
            #save_path = f'road_anomaly_detector/main/calibration/enoder_test_3.png'  # Modify this path as needed
            #save_image(image, save_path)

            height_px, width_px, _ = paper_size(image, lower=235, dispay=True, blurEn=False, morphEn=False)

            height_mm = height_px * cam.getSpartial()
            width_mm = width_px * cam.getSpartial()

            error_height = (abs(height_mm - a3_height_mm) / a3_height_mm) * 100
            error_width = (abs(width_mm - a3_width_mm) / a3_width_mm) * 100

            new_resolution = initial_resolution + (initial_resolution * error_height / 100)
            initial_resolution = new_resolution

            print(f"\nPixels: {height_px} x {width_px}")
            print(f"Real size [mm][h x w]: {height_mm:.2f} x {width_mm:.2f}")
            print(f"A3 paper  [mm][h x w]: {a3_height_mm:.2f} x {a3_width_mm:.2f}")
            print(f"Error [%]: {error_height:.2f}% x {error_width:.2f}%")
            print(f"Suggested new resolution: {new_resolution:.2f}\n")

            if input("Type 'q' to quit, or any other key to recalibrate: ").lower() == 'q':
                # camera.cleanup()
                break

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
