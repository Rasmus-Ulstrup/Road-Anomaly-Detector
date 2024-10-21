import matplotlib.pyplot as plt
import numpy as np

class camProbities:
    def __init__(self, focal, WD, CamSpecs):
        # Assign
        self.focal = focal  # Assuming focal length is in mm
        self.WD = WD  # Working distance in meters
        # Unpack the specs
        self.resolution_h, self.resolution_v = CamSpecs.get("Resolution", (0, 0))
        self.pixel_size_h, self.pixel_size_v = CamSpecs.get("Pixel Size", (0.0, 0.0))  # Assuming µm
        self.line_rate = CamSpecs.get("Line Rate", 0)
        self.wheel_diameter = CamSpecs.get("Wheel Diameter", 0)

    def getFOV(self):
        # Convert focal length to meters from mm, same for pixel width
        focal_length_m = self.focal / 1000  # Convert mm to meters
        pixel_width_m = self.pixel_size_h / 1e6  # Convert µm to meters

        # Calculate sensor width
        sensor_width_m = self.resolution_h * pixel_width_m

        # Calculate FOV
        FOV = (sensor_width_m / focal_length_m) * self.WD
        return FOV

    def getSpartial(self):
        # Calculate spatial resolution in meters per pixel
        FOV = self.getFOV()
        spatial_resolution = FOV / self.resolution_h
        return spatial_resolution * 1000  # Return in mm/pixel

    def getMaxSpeed(self):
        # Calculate maximum speed in m/s then convert to km/h
        spatial_resolution = self.getSpartial()
        max_speed_m_per_s = spatial_resolution * self.line_rate
        max_speed_km_per_h = max_speed_m_per_s * 3.6
        return max_speed_km_per_h
    
    def calculateLineRate(self, speed_kmh):
        # Convert speed from km/h to m/s
        speed_ms = speed_kmh * 1000 / 3600

        # Calculate spatial resolution (in meters per pixel)
        spatial_resolution_m = self.getSpartial() / 1000  # Convert mm to meters

        # Calculate line rate in lines per second
        line_rate = speed_ms / spatial_resolution_m
        return line_rate
    
    def calculateMaxExposureTime(self, speed_kmh):
        # Convert speed from km/h to m/s
        speed_ms = speed_kmh * 1000 / 3600

        # Calculate spatial resolution (in meters per pixel)
        spatial_resolution_m = self.getSpartial() / 1000  # Convert mm to meters

        # Maximum exposure time to avoid motion blur (in seconds)
        max_exposure_time = spatial_resolution_m / speed_ms
        return max_exposure_time * 1e6  # Return in milliseconds
    
    def calculateEncoderResolution(self):
        # Calculate circumference of the wheel
        circumference = np.pi * self.wheel_diameter  # In meters

        # Convert spatial resolution to meters per pixel
        spatial_resolution_m = self.getSpartial() / 1000  # Convert mm to meters

        # Calculate required pulses per revolution (PPR) for encoder
        encoder_resolution_ppr = circumference / spatial_resolution_m
        return encoder_resolution_ppr

    def __str__(self):
        return (f"Resolution: {self.resolution_h} x {self.resolution_v} pixels\n"
                f"Pixel Size: {self.pixel_size_h} x {self.pixel_size_v} µm\n"
                f"Line Rate: {self.line_rate} lines/s\n"
                f"FOV: {self.getFOV()} m\n"
                f"Spatial resolution: {self.getSpartial()} mm/pixel\n"
                f"Encoder Resolution: {self.calculateEncoderResolution()} pulses per revolution Quadture mode\n")
