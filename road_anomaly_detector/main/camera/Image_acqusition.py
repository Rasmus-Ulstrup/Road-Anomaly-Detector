import pypylon.pylon as py
import numpy as np
import cv2
from tkinter import Tk, TclError, filedialog
import os
import threading
import statistics
import queue
class LineScanCamera:
    #init function with the camera settings defined (changed in the instance)
    def __init__(self, frame_height=1333, exposure=20, trigger='encoder', compression='png', gamma=1):
        self.VIRTUAL_FRAME_HEIGHT = round(frame_height)
        self.trigger = trigger
        self.compression = compression
        self.exposure = exposure
        self.gamma = gamma


        # Initialize camera
        self.cam = self.initialize_camera()
        
        # Configure camera
        self.configure_camera()

        # Initialize the image and missing line placeholders
        self.img = np.ones((self.VIRTUAL_FRAME_HEIGHT, self.cam.Width.Value), dtype=np.uint8)
        self.missing_line = np.ones((1, self.cam.Width.Value), dtype=np.uint8) * 255  # Scanline height set to 1

        # Flag to stop capture loop
        self.stop_capture = False
        self.current_row = 0  # Keep track of current row being filled
        
        # Initialize queue and saver thread attributes
        self.image_queue = queue.Queue()  # Queue to hold images to be saved from RAM to HDD
        self.saver_thread = threading.Thread(target=self.save_images) #Thread to save the images
        self.saver_thread.daemon = True  # Daemonize thread to exit with the main program
        self.saver_thread.start()

        self.output_folder = '/media/driveA/test/deep_learning_images' #Set the output folder where images are to be stored
        self.image_count=0
        
        
    def image_length_mode(self):
        try:
            print("!Remeber in this mode, variable frame_height is spatial resulotion for 1 meter!")
            image_length_meters = float(input("Enter the image length (in meters): "))
            self.VIRTUAL_FRAME_HEIGHT = int(image_length_meters * self.VIRTUAL_FRAME_HEIGHT)  # Assuming 1 meter = 1333 scanlines
        except ValueError:
            print("Invalid input. Using default length of 1 meter.")
            self.VIRTUAL_FRAME_HEIGHT = 1333  # Default 1 meter with spatial resolution = 0.75

    def setup_output_folder(self):
        try:
            print("Select a folder to save the captured image:")
            Tk().withdraw()  # Hide the root window
            folder_name = input("Enter a folder name: ")
            self.output_folder = os.path.join("/home/crackscope/Road-Anomaly-Detector/test", folder_name)
            os.makedirs(self.output_folder, exist_ok=True)  # Create the folder if it doesn't already exist
        except TclError:
            print("Tkinter not available, please enter the path manually:")
            folder_name = input("Enter a folder name: ")
            self.output_folder = os.path.join("/home/crackscope/Road-Anomaly-Detector/test", folder_name)
            os.makedirs(self.output_folder, exist_ok=True)  # Create the folder if it doesn't already exist

        self.output_path = os.path.join(self.output_folder, f"captured_image.{self.compression}")
        print(f"Image will be saved to: {self.output_path}")
        
        # Create subfolders for organized storage
        self.processed_folder = os.path.join(self.output_folder, 'processed')
        os.makedirs(self.processed_folder, exist_ok=True)

        print(f"Processed images will be saved to: {self.processed_folder}")

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
        self.cam.PixelFormat.Value = "Mono8"  # Set to monochrome 8-bit format
        self.cam.Gain.Value = 1
        self.cam.ExposureTime.Value = self.exposure
        self.cam.BslShadingCorrectionSelector.Value = "PRNU"
        self.cam.BslShadingCorrectionMode.Value = "User"
        self.cam.Gamma.Value = self.gamma
        
        # Enable trigger based on the parameter
        if self.trigger == 'encoder':
            self.cam.TriggerSelector.Value = "LineStart"
            self.cam.TriggerSource.Value = "Line2"
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
        for idx in range(self.VIRTUAL_FRAME_HEIGHT): #size of image defined by virtual_frame_height
            with self.cam.RetrieveResult(20000) as result:
                if result.GrabSucceeded():
                    with result.GetArrayZeroCopy() as out_array:
                        self.img[idx] = out_array
                else:
                    self.img[idx] = self.missing_line
                    print(f"Missing line at index {idx}")

        self.cam.StopGrabbing()
        return self.img
    
    def capture_image_machine_learning(self, correction):
        self.cam.StartGrabbing()
        self.output_folder = '/media/driveA/test/deep_learning_images'
        
        # Create a separate thread to listen for user keyboard interrupt
        input_thread = threading.Thread(target=self.wait_for_stop_signal)
        input_thread.start()

        print("Capturing images for machine learning. Press ENTER to stop capturing.")
        
        # Create an empty list to store image lines for each image
        current_image_lines = []
        self.image_count = 0  # Counter for saved images
        self.current_row = 1
        old_array = 0
        # Capture loop
        while not self.stop_capture:
            with self.cam.RetrieveResult(20000) as result:
                if result.GrabSucceeded():
                    # Use GetArray() to safely retrieve the image line data
                    out_array = result.GetArray()
                    if (correction):
                        if self.current_row % 2:
                            curret_array =  out_array - old_array
                            current_image_lines.append(curret_array)
                        else:
                            old_array = out_array
                    if (correction==False):
                        if self.current_row % 2:
                            current_image_lines.append(out_array)
                    current_image_lines.append(out_array)
                    # Check if we have enough lines to form a complete image
                    if self.current_row == self.VIRTUAL_FRAME_HEIGHT:
                        # Stack the lines to form an image
                        captured_image = np.vstack(current_image_lines)

                        # Save the image
                        output_path = os.path.join(self.output_folder, f"vej_2_25_{self.image_count:05d}.{self.compression}")
                        self.image_count+=1
                        # Enqueue the image for saving
                        self.image_queue.put((captured_image, output_path, self.image_count))

                        # Clear the current image lines to start a new image
                        current_image_lines = []
                        self.current_row = 0
                else:
                    print(f"Missing line at row {self.current_row}")

                self.current_row += 1  # Move to the next row

        self.cam.StopGrabbing()
        self.image_queue.put(None)
        input_thread.join()  # Ensure input thread completes
        self.saver_thread.join()

        print("Capture stopped.")
    
 
    def save_images(self):
            #Thread for saving images, to ensure synchronosity in program
            while True:
                item = self.image_queue.get()
                if item is None:
                    break
                captured_image, output_path, self.image_count = item
                try:
                    cv2.imwrite(output_path, captured_image)
                    print(f"Saved image {self.image_count} at {output_path}")
                    
                except Exception as e:
                    print(f"Failed to save image {self.image_count}: {e}")
                finally:
                    self.image_queue.task_done()
    def capture_image_dynamic(self):
        self.cam.StartGrabbing()

        # Create a separate thread to lisen for user keyboard interrupts
        input_thread = threading.Thread(target=self.wait_for_stop_signal)
        input_thread.start()

        print("Capturing dynamic image. Press ENTER to stop capturing and save the image.")
        
        # Create an empty list to store image lines
        image_list = []

        while not self.stop_capture:
            with self.cam.RetrieveResult(20000) as result:
                if result.GrabSucceeded():
                    # Use GetArray() to safely retrieve the image line data
                    out_array = result.GetArray()
                    # Add the image line to the list
                    image_list.append(out_array)
                else:
                    # If grab failed, use the placeholder missing line
                    image_list.append(self.missing_line)
                    print(f"Missing line at row {self.current_row}")

                self.current_row += 1  # Move to the next row of the stitched image

        self.cam.StopGrabbing()
        input_thread.join()  # Ensure input thread completes

        self.img = np.vstack(image_list)

        # Return the Image in 2048x2048 format
        return self.img
    def capture_image_exposure_correction(self,correction):
        #PID variables
        target_brightness = 120.0
        Kp = 0.1
        Ki = 0.0001
        Kd = 0.005

        integral_error = 0.0
        previous_error = 0.0
        exposure_min = 2
        exposure_max = 10000

        self.cam.StartGrabbing()
        #Folder path for image storing
        self.output_folder = '/media/driveA/test/deep_learning_images' 

        # Prepare CSV logging
        csv_filename = os.path.join(self.output_folder, "pid_log.csv")
        file_exists = os.path.isfile(csv_filename)
        
        # Open the CSV in append mode once before the loop.
        with open(csv_filename, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            # If the file didn't exist, write a header row with relevant parameters
            if not file_exists:
                writer.writerow(["PID_error",
                                "New_exposure_time",
                                "Median_brightness",
                                "Target_brightness"])

            # Create a separate thread to listen for user input
            input_thread = threading.Thread(target=self.wait_for_stop_signal)
            input_thread.start()
            
            brightness_buffer = []
            image_list = []
            images_processed = 0
            print("Capturing images for machine learning. Press ENTER to stop capturing.") #Print to let the user know interrupt is available
            
            # Create an empty list to store image lines for each image
            current_image_lines = []
            self.image_count = 0  # Counter for saved images
            self.current_row = 1
            old_array = 0

            # Capture loop
            while not self.stop_capture:
                with self.cam.RetrieveResult(20000) as result:
                    if result.GrabSucceeded():
                        # Use GetArray() to safely retrieve the scanline data
                        out_array = result.GetArray()
                        
                        if correction:
                            if self.current_row % 2 == 1:
                                curret_array = out_array - old_array
                                current_image_lines.append(curret_array)
                            else:
                                old_array = out_array
                        else:
                            if self.current_row % 2 == 1:
                                current_image_lines.append(out_array)

                        current_image_lines.append(out_array)
                        
                        brightness_buffer.append(out_array)
                        images_processed += 1  # Increment the processed images count
                        
                        # Perform exposure update every predefined buffer size with PID
                        if images_processed % 10 == 0:  #Set to 10 for test, ideally increased further to account for anomalies
                            # compute the median brightness across the predefined  buffer
                            stacked_images = np.array(brightness_buffer)
                            median_brightness = np.median(stacked_images)
                            
                            print(f"Median brightness (last 10): {median_brightness}")
                            print(f"Current Exposure Time: {self.cam.ExposureTime.Value}")

                            # ──────────────────────────────────────────────────────────
                            # PID EXPOSURE CONTROL
                            # ──────────────────────────────────────────────────────────
                            error = target_brightness - median_brightness
                            integral_error += error
                            derivative_error = error - previous_error
                            pid_output = (Kp * error) + (Ki * integral_error) + (Kd * derivative_error)
                            print(f"PID error (pid_output): {pid_output}")

                            # Calculate new exposure time and clamp
                            new_exposure_time = self.cam.ExposureTime.Value + pid_output
                            new_exposure_time = max(exposure_min, min(new_exposure_time, exposure_max))
                            self.cam.ExposureTime.Value = new_exposure_time

                            # Write a row to CSV
                            writer.writerow([
                                pid_output,          # PID output (overall correction)
                                new_exposure_time,   # New Exposure Time
                                median_brightness,   # Median Brightness
                                target_brightness    # Target Brightness
                            ])

                            # Save current error for next loop
                            previous_error = error
                            # Reset for next batch
                            images_processed = 0
                            brightness_buffer = []

                        # Once enough lines have been captured to form a full image
                        if self.current_row == self.VIRTUAL_FRAME_HEIGHT:
                            # Stack the lines to form an image
                            captured_image = np.vstack(current_image_lines)

                            # Save the image
                            output_path = os.path.join(
                                self.output_folder,
                                f"test_{self.image_count:05d}.{self.compression}"
                            )
                            self.image_count += 1
                            # Enqueue the image for saving
                            self.image_queue.put((captured_image, output_path, self.image_count))

                            # Clear for the next image
                            current_image_lines = []
                            self.current_row = 0
                    else:
                        print(f"Missing line at row {self.current_row}")

                    self.current_row += 1  # Move to the next row

            # Stop grabbing
            self.cam.StopGrabbing()
            self.image_queue.put(None)
            input_thread.join()  # Ensure input thread completes
            self.saver_thread.join() #Thread for saving images, in 2048x2048 format
        
            print("Capture stopped.")
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
    # Create instance of the LineScanCamera class with desired camera at format settings
    camera = LineScanCamera(frame_height=2048, exposure=25, trigger='encoder', compression='png')

    #Set length mode:
    #camera.image_length_mode()
    
    #camera.capture_image(True)
    #camera.capture_image_dynamic()
    #camera.capture_image_dynamic_auto(True)
    #camera.capture_image_machine_learning(True)
    camera.capture_image_exposure_correction(False)
    #camera.capture_and_process_images()
    #camera.show_image()  # Optional: Display the image
    #camera.save_image()
    
    # Cleanup the resources pypylon function
    camera.cleanup()

if __name__ == "__main__":
    main()