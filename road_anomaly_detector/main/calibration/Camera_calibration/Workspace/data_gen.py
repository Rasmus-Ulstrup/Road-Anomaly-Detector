# Camera intrinsic parameters
fx = 1260  # Focal length in x (pixels)
u0 = 1020  # Principal point x-coordinate (pixels)
image_width = 2048  # Image width (pixels)

# Define two different camera poses (orientations)
poses = [
    {'rx_deg': 0, 'ry_deg': 0, 'rz_deg': 0, 'tx': 0, 'ty': 0, 'tz': 3000},
    {'rx_deg': 5, 'ry_deg': -2, 'rz_deg': 0, 'tx': 50, 'ty': -20, 'tz': 3000}
]

# Function to simulate capturing the pattern with the camera
def simulate_line_scan_camera(pose, vertical_lines, slanted_lines):
    rx_rad = np.deg2rad(pose['rx_deg'])
    ry_rad = np.deg2rad(pose['ry_deg'])
    rz_rad = np.deg2rad(pose['rz_deg'])
    tx, ty, tz = pose['tx'], pose['ty'], pose['tz']

    # Rotation matrices
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
    R = Rz @ Ry @ Rx
    t = np.array([[tx], [ty], [tz]])

    # Simulate the line-scan camera capturing feature points
    feature_points = []

    # Simulate viewing the pattern's vertical and slanted lines
    # For simplicity, we'll assume the camera sees the intersections of its viewing plane with the pattern lines

    # The camera's viewing plane is defined by its position and orientation
    # For a line-scan camera, we can simulate its field of view as a line in 3D space

    # For each vertical line, compute its intersection with the camera's viewing plane
    # In practice, this requires solving for the intersection of lines in 3D space

    # For this synthetic simulation, we'll create feature points along a line in space

    # Define a line of sight for the camera in the camera coordinate system
    num_feature_points = 20
    line_length_cam = 2000  # Length of the line in camera coordinates
    y_cam = np.linspace(-line_length_cam / 2, line_length_cam / 2, num_feature_points)
    x_cam = np.zeros_like(y_cam)
    z_cam = np.zeros_like(y_cam)

    # Transform the line of sight into the world coordinate system
    line_of_sight_world = R.T @ np.vstack((x_cam, y_cam, z_cam)) - R.T @ t

    # Find intersections with the pattern lines
    # For simplicity, we'll project these points onto the pattern plane (Z=0)
    Z = 0  # Pattern is in Z=0 plane
    scale = (Z - t[2]) / (R[2, :] @ line_of_sight_world)
    points_world = t + R @ (scale * line_of_sight_world)

    # Project the points onto the image plane
    X_cam = R @ (points_world - t)
    x_norm = X_cam[0, :] / X_cam[2, :]
    u = fx * x_norm + u0

    # Store the image coordinates (u) of the feature points
    return u

# Simulate capturing the pattern from two orientations
image_points_list = []
for pose in poses:
    u = simulate_line_scan_camera(pose, vertical_lines, slanted_lines)
    image_points_list.append(u)

    # Plot the simulated line-scan image
    plt.figure(figsize=(8, 2))
    plt.scatter(u, np.zeros_like(u), color='red', marker='|', s=100)
    plt.xlim(0, image_width)
    plt.title(f"Simulated Line-Scan Image (Pose: {pose})")
    plt.xlabel('u (pixels)')
    plt.yticks([])
    plt.show()
