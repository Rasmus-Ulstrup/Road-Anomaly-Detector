import pypylon.pylon as py
import numpy as np
import cv2

# Constants
SCANLINE_HEIGHT = 1

# Function to save the image
def save_image(image):
    save_prompt = input("Would you like to save the image? (y/n): ").strip().lower()
    if save_prompt == 'y':
        file_name = input("Enter the file name (with extension, e.g., image.png): ").strip()
        cv2.imwrite(file_name, image)
        print(f"Image saved as {file_name}")
    else:
        print("Image not saved.")

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
cam.ExposureTime.Value = 40

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

# Create a 2D image array with an arbitrary large initial height
initial_height = 1000  # Start with 1000 scanlines
img = np.ones((initial_height, cam.Width.Value), dtype=np.uint8)
missing_line = np.ones((SCANLINE_HEIGHT, cam.Width.Value), dtype=np.uint8) * 255  # Placeholder for missing scanlines

print("Waiting for trigger...")

# Start capturing scanlines
idx = 0
while True:
    # Dynamically resize the image array if needed
    if idx >= img.shape[0]:
        img = np.vstack((img, np.ones((1000, cam.Width.Value), dtype=np.uint8)))  # Increase height by 1000 scanlines

    # Capture one frame
    with cam.RetrieveResult(20000) as result:
        if result.GrabSucceeded():
            # Correctly place the captured scanline in the image array
            with result.GetArrayZeroCopy() as out_array:
                img[idx * SCANLINE_HEIGHT:idx * SCANLINE_HEIGHT + SCANLINE_HEIGHT] = out_array
        else:
            # Fill in with the missing scanline placeholder
            img[idx * SCANLINE_HEIGHT:idx * SCANLINE_HEIGHT + SCANLINE_HEIGHT] = missing_line
            print(f"Missing line at index {idx}")

    idx += 1

    # Check for key press on every iteration
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update display every 5000 scanlines
    if idx % 5000 == 0:
        cv2.imshow('Linescan View', img[:idx * SCANLINE_HEIGHT])  # Display only the filled part

# Display the final image
cv2.imshow('Linescan View', img[:idx * SCANLINE_HEIGHT])  # Display only the part that has been filled
print("Press a key to close....")
cv2.waitKey(0)  # Wait indefinitely until a key is pressed

# Optionally save the image
save_image(img[:idx * SCANLINE_HEIGHT])

# Cleanup
cam.StopGrabbing()
cam.Close()
cv2.destroyAllWindows()
