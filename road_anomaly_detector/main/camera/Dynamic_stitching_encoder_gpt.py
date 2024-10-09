import pypylon.pylon as py
import numpy as np
import cv2
import threading

# Global flag to signal when to stop the loop
stop_capture = False

# Function to listen for 'q' key press in a separate thread
def listen_for_keypress():
    global stop_capture
    input("Press 'q' and then Enter to stop capturing...\n")
    stop_capture = True

# Function to save the image
def save_image(image):
    save_prompt = input("Would you like to save the image? (y/n): ").strip().lower()
    if save_prompt == 'y':
        file_name = input("Enter the file name (with extension, e.g., image.png): ").strip()
        cv2.imwrite(file_name, image)
        print(f"Image saved as {file_name}")
    else:
        print("Image not saved.")

# Function to configure and open the camera
def setup_camera():
    # Initialize GigE camera
    tl_factory = py.TlFactory.GetInstance()

    cam = None
    for dev_info in tl_factory.EnumerateDevices():
        if dev_info.GetDeviceClass() == 'BaslerGigE':
            print("Using %s @ %s" % (dev_info.GetModelName(), dev_info.GetIpAddress()))
            cam = py.InstantCamera(tl_factory.CreateDevice(dev_info))
            break
    else:
        raise EnvironmentError("No GigE device found")

    # Open and configure camera
    cam.Open()
    cam.Height.Value = 1  # SCANLINE_HEIGHT
    cam.Width.Value = cam.Width.Max
    cam.PixelFormat.Value = "Mono8"  # Set to monochrome format
    cam.Gain.Value = 1
    cam.ExposureTime.Value = 40

    # Enable hardware trigger
    cam.TriggerSelector.Value = "LineStart"
    cam.TriggerSource.Value = "Line1"
    cam.TriggerMode.Value = "On"
    cam.TriggerActivation.Value = "RisingEdge"
    print("TriggerSource", cam.TriggerSource.Value)
    print("TriggerMode", cam.TriggerMode.Value)
    print("AcquisitionMode", cam.AcquisitionMode.Value)
    
    return cam

# Function to run the image acquisition in a separate thread
def capture_images(cam):
    global stop_capture
    cam.StartGrabbing()

    # Create a 2D image array with an arbitrary large initial height
    initial_height = 1000  # Start with 1000 scanlines
    img = np.ones((initial_height, cam.Width.Value), dtype=np.uint8)
    missing_line = np.ones((1, cam.Width.Value), dtype=np.uint8) * 255  # Placeholder for missing scanlines

    print("Waiting for trigger...")

    # Start capturing scanlines
    idx = 0
    while not stop_capture:
        # Dynamically resize the image array if needed
        if idx >= img.shape[0]:
            img = np.vstack((img, np.ones((1000, cam.Width.Value), dtype=np.uint8)))  # Increase height by 1000 scanlines

        # Capture one frame
        with cam.RetrieveResult(20000) as result:
            if result.GrabSucceeded():
                # Correctly place the captured scanline in the image array
                with result.GetArrayZeroCopy() as out_array:
                    img[idx * 1:idx * 1 + 1] = out_array
            else:
                # Fill in with the missing scanline placeholder
                img[idx * 1:idx * 1 + 1] = missing_line
                print(f"Missing line at index {idx}")

        idx += 1

    # Stop camera grabbing after loop ends
    cam.StopGrabbing()

    # Display the final image once the loop is stopped
    cv2.imshow('Linescan View', img[:idx * 1])  # Display only the part that has been filled
    print("Press a key to close....")
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed

    # Optionally save the image
    save_image(img[:idx * 1])

    # Cleanup
    cam.Close()
    cv2.destroyAllWindows()

# Main flow
if __name__ == "__main__":
    # Setup the camera
    camera = setup_camera()

    # Start the keypress listener thread
    keypress_thread = threading.Thread(target=listen_for_keypress)
    keypress_thread.start()

    # Start the image capture thread
    capture_thread = threading.Thread(target=capture_images, args=(camera,))
    capture_thread.start()

    # Wait for both threads to complete
    keypress_thread.join()
    capture_thread.join()
