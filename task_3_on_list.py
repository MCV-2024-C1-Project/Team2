import cv2
import numpy as np
import os


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
def create_background_model(image, bg_value=20):
    height, width, _ = image.shape

    # Get the pixels from the specified number of pixels from the edges
    top_strip = image[:bg_value, :, :]          # Top bg_value rows (full width)
    bottom_strip = image[-bg_value:, :, :]      # Bottom bg_value rows (full width)
    left_strip = image[:, :bg_value, :]         # Left bg_value columns (full height)
    right_strip = image[:, -bg_value:, :]       # Right bg_value columns (full width)

    # Compute the average color of the top-bottom and left-right regions separately
    avg_color_top_bottom = np.mean(np.vstack((top_strip, bottom_strip)), axis=(0, 1))
    avg_color_left_right = np.mean(np.hstack((left_strip, right_strip)), axis=(0, 1))

    # Combine the two averages to form the final background model
    avg_color_bg = (avg_color_top_bottom + avg_color_left_right) / 2

    return avg_color_bg


# Function to set the foreground square to white (255)
def set_edge_foreground(mask, start_x, end_x, start_y, end_y):
    mask[start_y:end_y, start_x:end_x] = 255  # Set the foreground square to white
    return mask


# Function to classify the image in RGB, HSV, and LAB color spaces
def classify_in_multiple_color_spaces(image, avg_color_bg_rgb, avg_color_bg_hsv, avg_color_bg_lab, threshold=50):
    height, width, _ = image.shape
    classified_mask_rgb = np.zeros((height, width), dtype=np.uint8)
    classified_mask_hsv = np.zeros((height, width), dtype=np.uint8)
    classified_mask_lab = np.zeros((height, width), dtype=np.uint8)

    # Convert image to HSV and LAB color spaces
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

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


# Function to evaluate the generated mask against the ground truth mask
def evaluate_mask(generated_mask, ground_truth_mask):
    # Compute Intersection over Union (IoU)
    intersection = np.logical_and(generated_mask == 255, ground_truth_mask == 255).sum()
    union = np.logical_or(generated_mask == 255, ground_truth_mask == 255).sum()
    iou = intersection / union if union != 0 else 0
    return iou


# Process all images in the folder and accumulate IoU performance
def process_folder_and_evaluate(image_folder):
    total_iou_rgb = 0
    total_iou_hsv = 0
    total_iou_lab = 0
    num_images = 0

    # Iterate over all .jpg images in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            image_jpg_path = os.path.join(image_folder, filename)
            image_png_path = os.path.join(image_folder, filename.replace('.jpg', '.png'))

            # Load the image and the corresponding mask
            image_jpg = cv2.imread(image_jpg_path)
            image_png = cv2.imread(image_png_path, cv2.IMREAD_GRAYSCALE)

            if image_jpg is None or image_png is None:
                print(f"Error loading {filename}, skipping.")
                continue

            # Create foreground square boundaries
            start_x, end_x, start_y, end_y = create_foreground_square(image_jpg, percentage=10)

            # Create background color models for RGB, HSV, and LAB
            avg_color_bg_rgb = create_background_model(image_jpg, bg_value=50)
            avg_color_bg_hsv = create_background_model(cv2.cvtColor(image_jpg, cv2.COLOR_BGR2HSV), bg_value=50)
            avg_color_bg_lab = create_background_model(cv2.cvtColor(image_jpg, cv2.COLOR_BGR2LAB), bg_value=50)

            # Classify the image using RGB, HSV, and LAB color spaces
            classified_mask_rgb, classified_mask_hsv, classified_mask_lab = classify_in_multiple_color_spaces(
                image_jpg, avg_color_bg_rgb, avg_color_bg_hsv, avg_color_bg_lab
            )

            # Set the foreground square to white (255) in all masks
            classified_mask_rgb = set_edge_foreground(classified_mask_rgb, start_x, end_x, start_y, end_y)
            classified_mask_hsv = set_edge_foreground(classified_mask_hsv, start_x, end_x, start_y, end_y)
            classified_mask_lab = set_edge_foreground(classified_mask_lab, start_x, end_x, start_y, end_y)

            # Evaluate the classified masks against the ground truth mask
            iou_rgb = evaluate_mask(classified_mask_rgb, image_png)
            iou_hsv = evaluate_mask(classified_mask_hsv, image_png)
            iou_lab = evaluate_mask(classified_mask_lab, image_png)

            # Print evaluation results for the image
            print(f"Results for {filename}:")
            print(f"  IoU (RGB): {iou_rgb:.4f}")
            print(f"  IoU (HSV): {iou_hsv:.4f}")
            print(f"  IoU (LAB): {iou_lab:.4f}")

            # Save the classified masks
            # cv2.imwrite(os.path.join(image_folder, f'classified_{filename[:-4]}_rgb.png'), classified_mask_rgb)
            # cv2.imwrite(os.path.join(image_folder, f'classified_{filename[:-4]}_hsv.png'), classified_mask_hsv)
            # cv2.imwrite(os.path.join(image_folder, f'classified_{filename[:-4]}_lab.png'), classified_mask_lab)

            # Accumulate the IoU for overall performance
            total_iou_rgb += iou_rgb
            total_iou_hsv += iou_hsv
            total_iou_lab += iou_lab
            num_images += 1

    # Compute and print the average IoU performance over all images
    if num_images > 0:
        avg_iou_rgb = total_iou_rgb / num_images
        avg_iou_hsv = total_iou_hsv / num_images
        avg_iou_lab = total_iou_lab / num_images

        print("\nOverall Performance:")
        print(f"  Average IoU (RGB): {avg_iou_rgb:.4f}")
        print(f"  Average IoU (HSV): {avg_iou_hsv:.4f}")
        print(f"  Average IoU (LAB): {avg_iou_lab:.4f}")
    else:
        print("No valid images were processed.")


# Define folder containing the images
image_folder = 'datasets/qsd2_w1'
process_folder_and_evaluate(image_folder)
