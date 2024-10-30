import numpy as np
import matplotlib.pyplot as plt

# Calibration pattern parameters
num_vertical_lines = 58  # Number of vertical lines (P_i)
num_slanted_lines = num_vertical_lines - 1  # Number of slanted lines (D_i)
line_length = 1000  # Length of the lines in mm
line_spacing = 20   # Spacing between vertical lines (h) in mm

# Generate vertical lines (P_i)
vertical_lines = []
for i in range(1, num_vertical_lines + 1):
    x = i * line_spacing
    y_start = 0
    y_end = line_length
    vertical_lines.append(((x, y_start), (x, y_end)))

# Generate slanted lines (D_i)
slanted_lines = []
for i in range(1, num_slanted_lines + 1):
    x_start = i * line_spacing
    x_end = (i + 1) * line_spacing
    y_start = 0
    y_end = line_length
    slanted_lines.append(((x_start, y_start), (x_end, y_end)))

# Plot the calibration pattern
plt.figure(figsize=(10, 8))
for line in vertical_lines:
    plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 'b-')
for line in slanted_lines:
    plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 'r-')
plt.title('Calibration Pattern')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.axis('equal')
plt.grid(True)
plt.show()
