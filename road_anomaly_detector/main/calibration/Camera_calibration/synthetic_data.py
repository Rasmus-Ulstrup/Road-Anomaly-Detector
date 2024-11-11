import numpy as np
import matplotlib.pyplot as plt

class CameraProjection:
    def __init__(self, fx, fy, u0, v0, img_width, img_height, rx_deg, ry_deg, rz_deg, t):
        # Store camera intrinsic parameters and image resolution
        self.fx = fx
        self.fy = fy
        self.u0 = u0
        self.v0 = v0
        self.img_width = img_width
        self.img_height = img_height

        # Convert rotation angles to radians
        rx_rad = np.deg2rad(rx_deg)
        ry_rad = np.deg2rad(ry_deg)
        rz_rad = np.deg2rad(rz_deg)

        # Compute rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx_rad), -np.sin(rx_rad)],
                       [0, np.sin(rx_rad), np.cos(rx_rad)]])

        Ry = np.array([[np.cos(ry_rad), 0, np.sin(ry_rad)],
                       [0, 1, 0],
                       [-np.sin(ry_rad), 0, np.cos(ry_rad)]])

        Rz = np.array([[np.cos(rz_rad), -np.sin(rz_rad), 0],
                       [np.sin(rz_rad), np.cos(rz_rad), 0],
                       [0, 0, 1]])

        # Combined rotation matrix
        self.R = Rz @ Ry @ Rx
        # Translation vector
        self.t = t.reshape((3, 1))

    def project_points(self, noise_std_dev, num_points=41, point_distance=60):
        # Generate 3D points along a straight line in the pattern coordinate system
        total_length = point_distance * (num_points - 1)
        x_pattern = np.linspace(-total_length / 2, total_length / 2, num_points)
        y_pattern = np.zeros(num_points)
        z_pattern = np.zeros(num_points)
        points_pattern = np.vstack((x_pattern, y_pattern, z_pattern))  # Shape: (3, num_points)

        # Transform points to the camera coordinate system
        points_camera = self.R @ points_pattern + self.t  # Shape: (3, num_points)

        # Extract X, Y, Z coordinates
        X_cam = points_camera[0, :]
        Y_cam = points_camera[1, :]
        Z_cam = points_camera[2, :]

        # Perspective projection (normalized image coordinates)
        x_norm = X_cam / Z_cam
        y_norm = Y_cam / Z_cam

        # Apply camera intrinsic parameters to get pixel coordinates
        u = self.fx * x_norm + self.u0
        v = self.fy * y_norm + self.v0

        # Filter points that are within the image frame
        valid_indices = (u >= 0) & (u <= self.img_width) & (v >= 0) & (v <= self.img_height)
        u_valid = u[valid_indices]
        v_valid = v[valid_indices]

        # Noise
        u_valid += np.random.normal(0, noise_std_dev, size=u_valid.shape)
        v_valid += np.random.normal(0, noise_std_dev, size=v_valid.shape)

        return u_valid, v_valid

    def plot_points(self, u_valid, v_valid):
        # Plot the projected points on the image plane
        plt.figure(figsize=(8, 8))
        plt.scatter(u_valid, v_valid, color='red', marker='o')
        plt.xlim(0, self.img_width)
        plt.ylim(self.img_height, 0)  # In image coordinates, the origin is at the top-left corner
        plt.title('Projected Points on Image Plane')
        plt.xlabel('u (pixels)')
        plt.ylabel('v (pixels)')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Example usage
    fx = 1260  # Focal length in x (pixels)
    fy = 1260  # Focal length in y (pixels)
    u0 = 1020  # Principal point x-coordinate (pixels)
    v0 = 1020  # Principal point y-coordinate (pixels)
    img_width = 2048  # Image width (pixels)
    img_height = 2048  # Image height (pixels)

    rx_deg = 0    # Rotation around x-axis
    ry_deg = 0   # Rotation around y-axis
    rz_deg = 0    # Rotation around z-axis

    t = np.array([0, 0, 1800])  # Translation vector (mm)

    # Instantiate the class
    camera_projection = CameraProjection(fx, fy, u0, v0, img_width, img_height, rx_deg, ry_deg, rz_deg, t)

    # Project points and plot
    u_valid, v_valid = camera_projection.project_points(0)
    camera_projection.plot_points(u_valid, v_valid)
