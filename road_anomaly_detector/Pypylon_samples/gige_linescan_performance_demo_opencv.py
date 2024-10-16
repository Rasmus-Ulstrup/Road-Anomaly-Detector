import pypylon.pylon as py
import numpy as np
import cv2

# This sample has been adapted to work with a monochrome GigE camera

# Constants
SCANLINE_HEIGHT = 1
VIRTUAL_FRAME_HEIGHT = 100

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
cam.Height = SCANLINE_HEIGHT
cam.Width.Value = 1072
#cam.CenterX = True
#cam.CenterY = True
cam.PixelFormat = "Mono8"  # Set to monochrome format
cam.Gain = 1
cam.ExposureTime = 10000

# Start grabbing
cam.StartGrabbing()

# Create a 2D image array for the monochrome camera
img = np.ones((VIRTUAL_FRAME_HEIGHT, cam.Width.Value), dtype=np.uint8)
# Create a missing line placeholder (also 2D)
missing_line = np.ones((SCANLINE_HEIGHT, cam.Width.Value), dtype=np.uint8) * 255
image_idx = 0

while True:
    for idx in range(VIRTUAL_FRAME_HEIGHT // SCANLINE_HEIGHT):
        with cam.RetrieveResult(2000) as result:
            if result.GrabSucceeded():
                with result.GetArrayZeroCopy() as out_array:
                    img[idx * SCANLINE_HEIGHT:idx * SCANLINE_HEIGHT + SCANLINE_HEIGHT] = out_array
            else:
                img[idx * SCANLINE_HEIGHT:idx * SCANLINE_HEIGHT + SCANLINE_HEIGHT] = missing_line
                print(idx)

    # Display the resulting frame (no conversion needed for monochrome)
    cv2.imshow('Linescan View', img)

    image_idx += 1
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

# When everything done, release the capture
cam.StopGrabbing()
cam.Close()
cv2.destroyAllWindows()