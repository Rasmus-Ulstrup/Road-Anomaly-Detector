import pypylon.pylon as py
import numpy as np
import cv2
from tkinter import Tk, TclError, filedialog
import os

class LineScanCamera:
    SCANLINE_HEIGHT = 1
    VIRTUAL_FRAME_HEIGHT_DEFAULT = 3114  # 1 meter

    def __init__(self, trigger='encoder', frame_height=VIRTUAL_FRAME_HEIGHT_DEFAULT, compression='png'):
        self.trigger = trigger
        self.VIRTUAL_FRAME_HEIGHT = int(frame_height)  # Set frame height based on user input
        self.compression = compression  # Set the image compression format
        
        # Set up folder for saving captured images
        self.directory = "/home/crackscope/Road-Anomaly-Detector/Test"
        self.setup_output_folder()

        # Initialize camera
        self.cam = self.initialize_camera()

        # Configure camera based on trigger type
        self.configure_camera()

        # Initialize the image and missing line placeholders
        self.img = np.ones((self.VIRTUAL_FRAME_HEIGHT, self.cam.Width.Value), dtype=np.uint8)
        self.missing_line = np.ones((self.SCANLINE_HEIGHT, self.cam.Width.Value), dtype=np.uint8) * 255

    def setup_output_folder(self):
        try:
            # Try to use Tkinter if available
            print("Enter a folder name to save the captured image:")
            folder_name = input("Folder name: ")
            self.output_folder = os.path.join(self.directory, folder_name)
            
            # Create the folder if it doesn't exist
            os.makedirs(self.output_folder, exist_ok=True)

            self.output_path = os.path.join(self.output_folder, f"captured_image.{self.compression}")
            print(f"Image will be saved to: {self.output_path}")
        except Exception as e:
            print(f"Error while setting up output folder: {e}")
            self.output_folder = os.getcwd()  # Fallback to current working directory
            self.output_path = os.path.join(self.output_folder, f"captured_image.{self.compression}")


    def initialize_camera(self):
        tl_factory = py.TlFactory.GetInstance()

        for dev_info in tl_factory.EnumerateDevices():
            if dev_info.GetDeviceClass() == 'BaslerGigE':
                print(f"Using {dev_info.GetModelName()} @ {dev_info.GetIpAddress()}")
                cam = py.InstantCamera(tl_factory.CreateDevice(dev_info))
                cam.Open()
                return cam
        raise EnvironmentError("No GigE device found")

    def configure_camera(self):
        self.cam.Height.Value = self.SCANLINE_HEIGHT
        self.cam.Width.Value = self.cam.Width.Max
        self.cam.PixelFormat.Value = "Mono8"  # Set to monochrome format
        self.cam.Gain.Value = 1
        self.cam.ExposureTime.Value = 20

        # Configure trigger
        if self.trigger == 'encoder':
            self.cam.TriggerSelector.Value = "LineStart"
            self.cam.TriggerSource.Value = "Line1"
            self.cam.TriggerMode.Value = "On"
            self.cam.TriggerActivation.Value = "RisingEdge"
        else:
            # Configure for software trigger if specified
            #self.cam.TriggerSelector.Value = "FrameStart"
            #self.cam.TriggerSource.Value = "Software"
            #self.cam.TriggerMode.Value = "On"
            #self.cam.TriggerActivation.Value = "RisingEdge"
            None

        print("TriggerSource:", self.cam.TriggerSource.Value)
        print("TriggerMode:", self.cam.TriggerMode.Value)
        print("AcquisitionMode:", self.cam.AcquisitionMode.Value)

    def capture_image(self):
        self.cam.StartGrabbing()
        print("Waiting for trigger...")

        # Capture one frame
        for idx in range(self.VIRTUAL_FRAME_HEIGHT // self.SCANLINE_HEIGHT):
            with self.cam.RetrieveResult(20000) as result:
                if result.GrabSucceeded():
                    with result.GetArrayZeroCopy() as out_array:
                        self.img[idx * self.SCANLINE_HEIGHT:idx * self.SCANLINE_HEIGHT + self.SCANLINE_HEIGHT] = out_array
                else:
                    self.img[idx * self.SCANLINE_HEIGHT:idx * self.SCANLINE_HEIGHT + self.SCANLINE_HEIGHT] = self.missing_line
                    print(f"Missing line at index {idx}")

        self.cam.StopGrabbing()

    def show_image(self):
        mirrored_img = cv2.flip(self.img, 1)
        cv2.imshow('Linescan View', mirrored_img)
        print("Press a key to close....")
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed

    def save_image(self):
        cv2.imwrite(self.output_path, self.img)
        print(f"Image saved at {self.output_path}")

    def cleanup(self):
        self.cam.Close()
        cv2.destroyAllWindows()

def main():
    # Create instance of the LineScanCamera class with parameters
    #if trigger != encoder it will be software trigger
    camera = LineScanCamera(trigger='', frame_height=1, compression='png')
    
    # Capture and display the image
    camera.capture_image()
    #camera.show_image()
    camera.save_image()
    
    # Cleanup the resources
    camera.cleanup()

if __name__ == "__main__":
    main()
