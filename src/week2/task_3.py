import cv2
import numpy as np
import os


# Function to create a background model from the edges
def create_background_model(image, bg_value=20):
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


# Function to classify the image in HSV color space
def classify_in_hsv(image, avg_color_bg_hsv, threshold=50):
    height, width, _ = image.shape
    classified_mask_hsv = np.zeros((height, width), dtype=np.uint8)

    # Convert image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Classify each pixel based on distance to background in HSV space
    for y in range(height):
        for x in range(width):
            pixel_hsv = image_hsv[y, x]
            dist_to_bg_hsv = np.linalg.norm(pixel_hsv - avg_color_bg_hsv)
            classified_mask_hsv[y, x] = 255 if dist_to_bg_hsv > threshold else 0

    return classified_mask_hsv


# Function to evaluate the generated mask against the ground truth mask
def evaluate_mask(generated_mask, ground_truth_mask):
    # Compute Intersection over Union (IoU)
    intersection = np.logical_and(generated_mask == 255, ground_truth_mask == 255).sum()
    union = np.logical_or(generated_mask == 255, ground_truth_mask == 255).sum()
    iou = intersection / union if union != 0 else 0
    return iou


# Process only one image
def process_single_image(image_jpg_path, image_png_path):
    # Load the image and the corresponding mask
    image_jpg = cv2.imread(image_jpg_path)
    image_png = cv2.imread(image_png_path, cv2.IMREAD_GRAYSCALE)

    if image_jpg is None or image_png is None:
        print("Error loading the files.")
        return

    # Create background color model for HSV
    avg_color_bg_hsv = create_background_model(cv2.cvtColor(image_jpg, cv2.COLOR_BGR2HSV), bg_value=50)

    # Classify the image using HSV color space
    classified_mask_hsv = classify_in_hsv(image_jpg, avg_color_bg_hsv)

    # Evaluate the classified mask against the ground truth mask
    iou_hsv = evaluate_mask(classified_mask_hsv, image_png)

    # Print evaluation results
    print(f"Results for {os.path.basename(image_jpg_path)}:")
    print(f"  IoU (HSV): {iou_hsv:.4f}")

    # Save the classified mask
    # output_path = image_jpg_path.replace('.jpg', '_classified_hsv.png')
    # cv2.imwrite(output_path, classified_mask_hsv)
    # print(f"Classified mask saved at {output_path}")


# Define paths to the image and mask
image_jpg_path = 'datasets/qsd2_w1/00002.jpg'
image_png_path = 'datasets/qsd2_w1/00002.png'

# Process the single image
process_single_image(image_jpg_path, image_png_path)
