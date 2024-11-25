import numpy as np
from PIL import Image

# Define dimensions in millimeters and desired DPI
line_spacing = 25  # mm
line_width = 3  # mm
height_mm, width_mm = 200, 1525  # Pattern dimensions in mm
dpi = 1200  # Print quality

# Page size selection (A4 or A3)
page_height_mm, page_width_mm = 297, 420  # A4 in mm (change to 297, 420 for A3)
margin_mm = 5  # Define margin in millimeters

# Calculate pixels per mm and convert dimensions to pixels
pixels_per_mm = dpi / 25.4
print("Pixels per mm: ", pixels_per_mm)
height_px = int(height_mm * pixels_per_mm)
width_px = int(width_mm * pixels_per_mm)

# Page dimensions with margin
page_width_px = int((page_width_mm - 2 * margin_mm) * pixels_per_mm)
page_height_px = int((page_height_mm - 2 * margin_mm) * pixels_per_mm)
margin_px = int(margin_mm * pixels_per_mm)

# Calculate number of pages needed to cover the width
num_pages = int(np.ceil(width_px / page_width_px))

# Define the padding in mm and convert to pixels
padding_mm = 36  # Amount of white space to add below
padding_px = int(padding_mm * pixels_per_mm)

# Create the image with the pattern and add padding at the bottom
image = np.ones((height_px + padding_px, width_px), dtype=np.uint8) * 255  # White background
line_spacing_px = int(line_spacing * pixels_per_mm)  # Spacing for x
print("Line spacing PX: ", line_spacing_px)
line_width_px = int(line_width * pixels_per_mm)  # Line width for x
print("Line width PX: ", line_spacing_px)

vertical_line_count = 0  # Counter for vertical lines

# Draw vertical and diagonal lines, leaving padding at the bottom
for x in range(0, width_px - line_spacing_px, line_spacing_px):
    # Draw the vertical line centered within the x
    start_x = x + (line_spacing_px - line_width_px) // 2
    image[:height_px, start_x:start_x + line_width_px] = 0  # Vertical line
    vertical_line_count += 1

    # Only draw a diagonal line if it's not the last interval
    if x + 2 * line_spacing_px <= width_px:
        # Calculate the center of the next vertical line
        next_start_x = start_x + line_spacing_px - (line_width_px / 2)

        # Draw the diagonal line from the middle of the current line to the next
        for y in range(height_px):
            diag_x = int(start_x + (next_start_x - start_x) * (y / height_px) + line_width_px // 2)
            if diag_x < width_px:
                image[height_px - 1 - y, diag_x:diag_x + line_width_px] = 0

# Calculate the true distance between lines
true_distance_mm = line_spacing_px / pixels_per_mm

# Print results
print(f"Number of vertical lines: {vertical_line_count}")
print(f"True distance between lines: {true_distance_mm:.2f} mm")

# Save the full pattern as a single PNG
raw_pattern_image = Image.fromarray(image)
raw_pattern_image.save("road_anomaly_detector/main/calibration/Camera_calibration/raw_pattern.png", dpi=(dpi, dpi))

# Create an empty list to store the pages for the PDF
pages = []
for page in range(num_pages):
    # Slice the image for each page
    start_x = page * page_width_px
    end_x = min(start_x + page_width_px, width_px)
    page_image = image[:, start_x:end_x]

    # Convert the slice to a PIL Image and set DPI for true scaling
    pil_image = Image.fromarray(page_image)
    pil_image = pil_image.convert("L")  # Ensure grayscale mode

    # Create a larger canvas for A4 or A3 with margin
    canvas_width_px = page_width_px + 2 * margin_px
    canvas_height_px = height_px + padding_px + 2 * margin_px
    pdf_page = Image.new("L", (canvas_width_px, canvas_height_px), "white")
    
    # Paste the image slice in the center of the canvas
    pdf_page.paste(pil_image, (margin_px, margin_px))

    # Draw the indicator lines at the inner edges of the margins
    line_thickness_px = int(1 * pixels_per_mm)  # Line thickness for indicator lines
    for y in range(canvas_height_px):
        # Left margin indicator line
        pdf_page.putpixel((margin_px - line_thickness_px // 2, y), 0)  
        # Right margin indicator line
        pdf_page.putpixel((canvas_width_px - margin_px + line_thickness_px // 2 - 1, y), 0)

    # Append this page to the list with the correct DPI
    pages.append(pdf_page)

# Save all slices as individual pages in a single PDF file
pages[0].save("road_anomaly_detector/main/calibration/Camera_calibration/print_of_pattern.pdf", save_all=True, append_images=pages[1:], dpi=(dpi, dpi))

print(f"Multi-page PDF saved as 'print_of_pattern.pdf' with {num_pages} pages.")
