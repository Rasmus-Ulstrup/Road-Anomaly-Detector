import pypylon.pylon as py
import numpy as np
import cv2
from tkinter import Tk, TclError, filedialog
import os
import threading

class LineScanCamera:
    def __init__(self, frame_height=1557, exposure=20, trigger='encoder', compression='png', max_capture_meters=500):
        self.VIRTUAL_FRAME_HEIGHT = round(frame_height)  # Set from parameter
        self.trigger = trigger
        self.compression = compression
        self.exposure = exposure


        self.MAX_FRAME_HEIGHT = int(max_capture_meters * frame_height)  # Pre-allocate for multiple meters
        # Initialize camera
        self.cam = self.initialize_camera()
        
        # Configure camera
        self.configure_camera()

        # Initialize the image and missing line placeholders
        self.img = np.ones((self.VIRTUAL_FRAME_HEIGHT, self.cam.Width.Value), dtype=np.uint8)
        self.missing_line = np.ones((1, self.cam.Width.Value), dtype=np.uint8) * 255  # Scanline height is always 1

        # Flag to stop capture loop
        self.stop_capture = False
        self.current_row = 0  # Keep track of current row being filled

    def image_length_mode(self):
        try:
            print("!Remeber in this mode, variable frame_height is spatial resulotion for 1 meter!")
            image_length_meters = float(input("Enter the image length (in meters): "))
            self.VIRTUAL_FRAME_HEIGHT = int(image_length_meters * self.VIRTUAL_FRAME_HEIGHT)  # Assuming 1 meter = 1557 scanlines
        except ValueError:
            print("Invalid input. Using default length of 1 meter.")
            self.VIRTUAL_FRAME_HEIGHT = 1557  # Default 1 meter

    def setup_output_folder(self):
        try:
            print("Select a folder to save the captured image:")
            Tk().withdraw()  # Hide the root window
            folder_name = input("Enter a folder name: ")
            self.output_folder = os.path.join("/home/crackscope/Road-Anomaly-Detector/Test", folder_name)
            os.makedirs(self.output_folder, exist_ok=True)  # Create the folder if it doesn't exist
        except TclError:
            print("Tkinter not available, please enter the path manually:")
            folder_name = input("Enter a folder name: ")
            self.output_folder = os.path.join("/home/crackscope/Road-Anomaly-Detector/Test", folder_name)
            os.makedirs(self.output_folder, exist_ok=True)  # Create the folder if it doesn't exist

        self.output_path = os.path.join(self.output_folder, f"captured_image.{self.compression}")
        print(f"Image will be saved to: {self.output_path}")

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
        self.cam.Height.Value = 1  # Scanline height is always 1
        self.cam.Width.Value = self.cam.Width.Max
        self.cam.PixelFormat.Value = "Mono8"  # Set to monochrome format
        self.cam.Gain.Value = 1
        self.cam.ExposureTime.Value = self.exposure

        # Enable trigger based on the parameter
        if self.trigger == 'encoder':
            self.cam.TriggerSelector.Value = "LineStart"
            self.cam.TriggerSource.Value = "Line1"
            self.cam.TriggerMode.Value = "On"
            self.cam.TriggerActivation.Value = "RisingEdge"
        else:
            # self.cam.TriggerSelector.Value = "FrameStart"
            # self.cam.TriggerSource.Value = "Software"
            # self.cam.TriggerMode.Value = "Off"
            # self.cam.TriggerActivation.Value = "RisingEdge"
            None
        print("TriggerSource:", self.cam.TriggerSource.Value)
        print("TriggerMode:", self.cam.TriggerMode.Value)
        print("AcquisitionMode:", self.cam.AcquisitionMode.Value)

    def capture_image(self):
        self.cam.StartGrabbing()

        # Capture one frame
        for idx in range(self.VIRTUAL_FRAME_HEIGHT):
            with self.cam.RetrieveResult(20000) as result:
                if result.GrabSucceeded():
                    with result.GetArrayZeroCopy() as out_array:
                        self.img[idx] = out_array
                else:
                    self.img[idx] = self.missing_line
                    print(f"Missing line at index {idx}")

        self.cam.StopGrabbing()
        return self.img
    def capture_image_dynamic(self):
        self.cam.StartGrabbing()

        # Create a separate thread to listen for user input
        input_thread = threading.Thread(target=self.wait_for_stop_signal)
        input_thread.start()

        print("Capturing dynamic image. Press ENTER to stop capturing and save the image.")
        
        # Create an empty list to store scan lines
        image_list = []

        # Capture loop
        while not self.stop_capture:
            if self.current_row >= self.MAX_FRAME_HEIGHT:
                print("Reached maximum capture height.")
                break

            with self.cam.RetrieveResult(20000) as result:
                if result.GrabSucceeded():
                    # Use GetArray() to safely retrieve the scanline data
                    out_array = result.GetArray()
                    # Add the scanline to the list
                    image_list.append(out_array)
                else:
                    # If grab failed, use the placeholder missing line
                    image_list.append(self.missing_line)
                    print(f"Missing line at row {self.current_row}")

                self.current_row += 1  # Move to the next row

        self.cam.StopGrabbing()
        input_thread.join()  # Ensure input thread completes

        # Convert the list of captured lines into a NumPy array
        self.img = np.vstack(image_list)

    def wait_for_stop_signal(self):
        input()  # Wait for user to press ENTER
        self.stop_capture = True  # Set flag to stop the capture
        
    def show_image(self):
        self.img = cv2.flip(self.img,1)
        cv2.imshow('Linescan View', self.img)
        print("Press a key to close....")
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed

    def save_image(self):
        self.setup_output_folder()
        # Logic to handle file naming if the image already exists
        base_path = os.path.splitext(self.output_path)[0]
        count = 0
        while os.path.exists(self.output_path):
            count += 1
            self.output_path = f"{base_path}{count}.{self.compression}"
        self.img = cv2.flip(self.img,1)
        cv2.imwrite(self.output_path, self.img)
        print(f"Image saved at {self.output_path}")

    def cleanup(self):
        self.cam.Close()
        cv2.destroyAllWindows()

def main():
    # Create instance of the LineScanCamera class
    camera = LineScanCamera(frame_height=1557*5, exposure=10, trigger='encoder', compression='png')

    #Set length mode:
    #camera.image_length_mode()
    
    # Capture and display the image
    camera.capture_image()

    #camera.capture_image_dynamic()
    #camera.show_image()  # Optional: Display the image
    camera.save_image()
    
    # Cleanup the resources
    camera.cleanup()

if __name__ == "__main__":
    main()
