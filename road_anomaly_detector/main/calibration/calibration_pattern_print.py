import numpy as np
from PIL import Image

# Define dimensions in millimeters and desired DPI
height_mm, width_mm = 100, 1560  # Pattern dimensions in mm
dpi = 300  # Print quality

# Page size selection (A4 or A3)
page_height_mm, page_width_mm = 210, 297  # A4 in mm (change to 297, 420 for A3)
margin_mm = 5  # Define margin in millimeters

# Calculate pixels per mm and convert dimensions to pixels
pixels_per_mm = dpi / 25.4
height_px = int(height_mm * pixels_per_mm)
width_px = int(width_mm * pixels_per_mm)

# Page dimensions with margin
page_width_px = int((page_width_mm - 2 * margin_mm) * pixels_per_mm)
page_height_px = int((page_height_mm - 2 * margin_mm) * pixels_per_mm)
margin_px = int(margin_mm * pixels_per_mm)

# Calculate number of pages needed to cover the width
num_pages = int(np.ceil(width_px / page_width_px))

# Create the image with the pattern
image = np.ones((height_px, width_px), dtype=np.uint8) * 255  # White background
line_spacing_px = int(40 * pixels_per_mm)  # Spacing for 40 mm
line_width_px = int(2 * pixels_per_mm)  # Line width for 2 mm

# Draw vertical and diagonal lines
for x in range(0, width_px - line_spacing_px, line_spacing_px):
    # Draw the vertical line centered within the 40 mm interval
    start_x = x + (line_spacing_px - line_width_px) // 2
    image[:, start_x:start_x + line_width_px] = 0  # Vertical line

    # Only draw a diagonal line if it's not the last interval
    if x + 2 * line_spacing_px <= width_px:
        # Calculate the center of the next vertical line
        next_start_x = start_x + line_spacing_px - (line_width_px / 2)

        # Draw the diagonal line from the middle of the current line to the next
        for y in range(height_px):
            diag_x = int(start_x + (next_start_x - start_x) * (y / height_px) + line_width_px // 2)
            if diag_x < width_px:
                image[height_px - 1 - y, diag_x:diag_x + line_width_px] = 0

# Save the full pattern as a single PNG
raw_pattern_image = Image.fromarray(image)
raw_pattern_image.save("road_anomaly_detector/main/calibration/raw_pattern.png", dpi=(dpi, dpi))

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
    canvas_height_px = height_px + 2 * margin_px
    pdf_page = Image.new("L", (canvas_width_px, canvas_height_px), "white")
    
    # Paste the image slice in the center of the canvas
    pdf_page.paste(pil_image, (margin_px, margin_px))

    # Append this page to the list with the correct DPI
    pages.append(pdf_page)

# Save all slices as individual pages in a single PDF file
pages[0].save("road_anomaly_detector/main/calibration/print_of_pattern.pdf", save_all=True, append_images=pages[1:], dpi=(dpi, dpi))

print(f"Multi-page PDF saved as 'pattern_output_multi_page.pdf' with {num_pages} pages.")
