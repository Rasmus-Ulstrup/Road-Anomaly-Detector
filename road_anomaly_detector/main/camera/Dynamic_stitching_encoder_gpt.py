import pypylon.pylon as py
import numpy as np
import cv2
import threading
from pynput import keyboard
import os
# Global flag to signal when to stop the loop
stop_capture = False
key = False
# Function to listen for 'q' key press in a separate thread

def listen_for_keypress():
    global stop_capture
    input()
    stop_capture = True

# Function to save the image
def save_image(image):
    save_prompt = input("Would you like to save the image? (y/n): ").strip().lower()
    if save_prompt == 'y':
        directory = "/home/crackscope/Road-Anomaly-Detector/Test"

        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = input("Enter the file name (with extension, e.g., image.png): ").strip()
        file_path = os.path.join(directory,file_name)
        cv2.imwrite(file_path, image)
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
    cam.ExposureTime.Value = 20

    # Enable hardware trigger
    cam.TriggerSelector.Value = "LineStart"
    cam.TriggerSource.Value = "Line1"
    cam.TriggerMode.Value = "On"
    cam.TriggerActivation.Value = "RisingEdge"
    print("TriggerSource", cam.TriggerSource.Value)
    print("TriggerMode", cam.TriggerMode.Value)
    print("AcquisitionMode", cam.AcquisitionMode.Value)
    
    cam.PacketSize.SetValue(1500)  # Example packet size, adjust for your network
    cam.InterPacketDelay.SetValue(1000)  # Increase if necessary
    return cam

# Function to run the image acquisition in a separate thread
def capture_images(cam):
    global stop_capture
    cam.StartGrabbing()

    # Preallocate a large image array based on expected maximum lines
    max_lines = 100000  # Adjust this as necessary
    img = np.zeros((max_lines, cam.Width.Value), dtype=np.uint8)  # Initialize with zeros
    idx = 0

    print("Waiting for trigger...")

    while not stop_capture:
        print(f"Grabbing frame {idx}...")  # Log frame index
        with cam.RetrieveResult(20000) as result:
            if result.GrabSucceeded():
                print(f"Frame {idx} successfully grabbed.")
                with result.GetArrayZeroCopy() as out_array:
                    img[idx] = out_array
                idx += 1
            else:
                print(f"Frame {idx} grab failed.")

    cam.StopGrabbing()

    # Save or process the captured image
    final_image = img[:idx]  # Use only the valid part of the image
    mirrored_img = cv2.flip(final_image, 1)  # Flip if needed
    save_image(mirrored_img)

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
