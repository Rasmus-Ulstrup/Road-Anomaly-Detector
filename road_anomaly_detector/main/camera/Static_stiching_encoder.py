import pypylon.pylon as py
import numpy as np
import cv2

from tkinter import Tk
from tkinter.filedialog import askdirectory
import os
# Constants
SCANLINE_HEIGHT = 1
VIRTUAL_FRAME_HEIGHT = 3114 # 1 meter

# User input for image length in meters
try:
    image_length_meters = float(input("Enter the image length (in meters): "))
    VIRTUAL_FRAME_HEIGHT = int(image_length_meters * VIRTUAL_FRAME_HEIGHT)  # Assuming 1 meter = 3114 scanlines
except ValueError:
    print("Invalid input. Using default length of 1 meter.")
    VIRTUAL_FRAME_HEIGHT = 3114  # Default 1 meter

# Prompt user to select a folder to save the image
print("Select a folder to save the captured image:")
output_folder = askdirectory()
if not output_folder:
    print("No folder selected, saving to current directory.")
    output_folder = os.getcwd()  # Default to current working directory

# Image file path
output_path = os.path.join(output_folder, "captured_image.png")
print(f"Image will be saved to: {output_path}")

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
cam.Height.Value = SCANLINE_HEIGHT
cam.Width.Value = cam.Width.Max
cam.PixelFormat.Value = "Mono8"  # Set to monochrome format
cam.Gain.Value = 1
cam.ExposureTime.Value = 30

# Enable hardware trigger
cam.TriggerSelector.Value = "LineStart"
cam.TriggerSource.Value = "Line1"
cam.TriggerMode.Value = "On"
cam.TriggerActivation.Value = "RisingEdge"
print("TriggerSource", cam.TriggerSource.Value)
print("TriggerMode", cam.TriggerMode.Value)
print("AcquisitionMode", cam.AcquisitionMode.Value)

# Start grabbing
cam.StartGrabbing()

# Create a 2D image array for the monochrome camera
img = np.ones((VIRTUAL_FRAME_HEIGHT, cam.Width.Value), dtype=np.uint8)
# Create a missing line placeholder (also 2D)
missing_line = np.ones((SCANLINE_HEIGHT, cam.Width.Value), dtype=np.uint8) * 255

print("Waiting for trigger...")

# Capture one frame
for idx in range(VIRTUAL_FRAME_HEIGHT // SCANLINE_HEIGHT):
    with cam.RetrieveResult(20000) as result:
        if result.GrabSucceeded():
            with result.GetArrayZeroCopy() as out_array:
                img[idx * SCANLINE_HEIGHT:idx * SCANLINE_HEIGHT + SCANLINE_HEIGHT] = out_array
        else:
            img[idx * SCANLINE_HEIGHT:idx * SCANLINE_HEIGHT + SCANLINE_HEIGHT] = missing_line
            print(idx)

mirrored_img = cv2.flip(img,1)
# Display the resulting frame
cv2.imshow('Linescan View', mirrored_img)
print("Press a key to close....")
cv2.waitKey(0)  # Wait indefinitely until a key is pressed

# Cleanup
cam.StopGrabbing()
cam.Close()
cv2.destroyAllWindows()
