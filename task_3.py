import cv2
import numpy as np

# Load images
image_jpg_path = 'datasets/qsd2_w1/00002.jpg'
image_png_path = 'datasets/qsd2_w1/00002.png'

image_jpg = cv2.imread(image_jpg_path)
image_png = cv2.imread(image_png_path, cv2.IMREAD_GRAYSCALE)  # Assuming PNG is a binary mask


# Function to create a foreground square (10% of the height and width)
def create_foreground_square(image, percentage=10):
    height, width, _ = image.shape

    # Calculate the dimensions of the square (10% of height and width)
    square_height = int(height * (percentage / 100))
    square_width = int(width * (percentage / 100))

    # Calculate the center of the image
    center_x = width // 2
    center_y = height // 2

    # Calculate the coordinates of the square (centered)
    start_x = center_x - square_width // 2
    end_x = center_x + square_width // 2
    start_y = center_y - square_height // 2
    end_y = center_y + square_height // 2

    return start_x, end_x, start_y, end_y


# Function to create a background model from the edges
def create_background_model(image, bg_value=50):
    height, width, _ = image.shape

    # Get the pixels from the specified number of pixels from the edges
    top_strip = image[:bg_value, :, :]          # Top bg_value rows (full width)
    bottom_strip = image[-bg_value:, :, :]      # Bottom bg_value rows (full width)
    left_strip = image[:, :bg_value, :]         # Left bg_value columns (full height)
    right_strip = image[:, -bg_value:, :]       # Right bg_value columns (full height)

    # Compute the average color of the top-bottom and left-right regions separately
    avg_color_top_bottom = np.mean(np.vstack((top_strip, bottom_strip)), axis=(0, 1))
    avg_color_left_right = np.mean(np.hstack((left_strip, right_strip)), axis=(0, 1))

    # Combine the two averages to form the final background model
    avg_color_bg = (avg_color_top_bottom + avg_color_left_right) / 2

    return avg_color_bg


# Function to automatically set the edge pixels as background
def set_edge_background(mask, top, bottom, left, right):
    # Set top, bottom, left, and right edge pixels as background (0)
    mask[:top, :] = 0          # Top edge
    mask[-bottom:, :] = 0      # Bottom edge
    mask[:, :left] = 0         # Left edge
    mask[:, -right:] = 0       # Right edge


# New function: Set the foreground square to white (255)
def set_edge_foreground(mask, start_x, end_x, start_y, end_y):
    # Set the pixels within the square area to white (255)
    mask[start_y:end_y, start_x:end_x] = 255  # Foreground square

    return mask


# Function to apply morphological closing around the predefined foreground square
def apply_morphological_closing_around_foreground_square(mask, start_x, end_x, start_y, end_y, kernel_size=(5, 5)):
    # Create a copy of the mask to apply closing outside the square
    mask_outside_square = mask.copy()

    # Set the square area to 0 (to protect it from being modified)
    # mask_outside_square[start_y:end_y, start_x:end_x] = 0

    # Define the kernel for morphological operations
    # kernel = np.ones(kernel_size, np.uint8)

    # Apply morphological closing (dilation followed by erosion) outside the square
    # closed_outside_square = cv2.morphologyEx(mask_outside_square, cv2.MORPH_CLOSE, kernel)

    # Combine the original square with the processed outside area
    final_mask = mask_outside_square
    final_mask[start_y:end_y, start_x:end_x] = mask[start_y:end_y, start_x:end_x]  # Corrected line

    return final_mask


# Function to classify foreground and background in RGB, HSV, and LAB color spaces
def classify_in_multiple_color_spaces(image, avg_color_bg_rgb, start_x, end_x, start_y, end_y, threshold=50):
    height, width, _ = image.shape
    classified_mask_rgb = np.zeros((height, width), dtype=np.uint8)
    classified_mask_hsv = np.zeros((height, width), dtype=np.uint8)
    classified_mask_lab = np.zeros((height, width), dtype=np.uint8)

    # Convert image to HSV and LAB color spaces
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Convert background average color to HSV and LAB
    avg_color_bg_hsv = cv2.cvtColor(np.uint8([[avg_color_bg_rgb]]), cv2.COLOR_BGR2HSV)[0][0]
    avg_color_bg_lab = cv2.cvtColor(np.uint8([[avg_color_bg_rgb]]), cv2.COLOR_BGR2LAB)[0][0]

    # Classify each pixel based on distance to background in RGB, HSV, and LAB spaces
    for y in range(height):
        for x in range(width):
            pixel_rgb = image[y, x]
            pixel_hsv = image_hsv[y, x]
            pixel_lab = image_lab[y, x]

            # Calculate distance to background in RGB, HSV, and LAB
            dist_to_bg_rgb = np.linalg.norm(pixel_rgb - avg_color_bg_rgb)
            dist_to_bg_hsv = np.linalg.norm(pixel_hsv - avg_color_bg_hsv)
            dist_to_bg_lab = np.linalg.norm(pixel_lab - avg_color_bg_lab)

            # Classify the pixel based on proximity to background in each color space
            classified_mask_rgb[y, x] = 255 if dist_to_bg_rgb > threshold else 0
            classified_mask_hsv[y, x] = 255 if dist_to_bg_hsv > threshold else 0
            classified_mask_lab[y, x] = 255 if dist_to_bg_lab > threshold else 0

    return classified_mask_rgb, classified_mask_hsv, classified_mask_lab


# Parameters for edge background (based on earlier calculations)
top_bg = 20     # Example: 20 pixels from the top
bottom_bg = 20  # Example: 20 pixels from the bottom
left_bg = 20    # Example: 20 pixels from the left
right_bg = 20   # Example: 20 pixels from the right

# Create foreground square boundaries
start_x, end_x, start_y, end_y = create_foreground_square(image_jpg, percentage=10)

# Create background color model from RGB
avg_color_bg_rgb = create_background_model(image_jpg, bg_value=50)

# Classify the image using RGB, HSV, and LAB color spaces
classified_mask_rgb, classified_mask_hsv, classified_mask_lab = classify_in_multiple_color_spaces(
    image_jpg, avg_color_bg_rgb, start_x, end_x, start_y, end_y
)

# Set the foreground square to white (255) in all masks
classified_mask_rgb = set_edge_foreground(classified_mask_rgb, start_x, end_x, start_y, end_y)
classified_mask_hsv = set_edge_foreground(classified_mask_hsv, start_x, end_x, start_y, end_y)
classified_mask_lab = set_edge_foreground(classified_mask_lab, start_x, end_x, start_y, end_y)

# Save the classified binary masks for visualization
cv2.imwrite('classified_foreground_rgb.jpg', classified_mask_rgb)
cv2.imwrite('classified_foreground_hsv.jpg', classified_mask_hsv)
cv2.imwrite('classified_foreground_lab.jpg', classified_mask_lab)

print("Masks for RGB, HSV, and LAB saved.")
