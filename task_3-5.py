import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

from utils import find_most_similar_image, calculate_histogram, compare_histograms
# Function to create a background model from the edges

old_stdout = sys.stdout
log_file = open("Output.log", "w", encoding='utf-8')
sys.stdout = log_file


def create_background_model(image, bg_value=None):

    height, width, _ = image.shape

    # Calculate dynamic bg_value based on image dimensions if not provided
    if bg_value is None:
        bg_value = int(min(height, width) * 0.05)

    # Get the pixels from the specified number of pixels from the edges
    # Top bg_value rows (full width)
    top_strip = image[:bg_value, :, :]
    # Bottom bg_value rows (full width)
    bottom_strip = image[-bg_value:, :, :]
    # Left bg_value columns (full height)
    left_strip = image[:, :bg_value, :]
    # Right bg_value columns (full width)
    right_strip = image[:, -bg_value:, :]

    # Compute the average color of the top-bottom and left-right regions separately
    avg_color_top_bottom = np.mean(
        np.vstack((top_strip, bottom_strip)), axis=(0, 1))
    avg_color_left_right = np.mean(
        np.hstack((left_strip, right_strip)), axis=(0, 1))

    # Combine the two averages to form the final background model
    avg_color_bg = (avg_color_top_bottom + avg_color_left_right) / 2

    return avg_color_bg


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
            classified_mask_rgb[y,
                                x] = 255 if dist_to_bg_rgb > threshold else 0
            classified_mask_hsv[y,
                                x] = 255 if dist_to_bg_hsv > threshold else 0
            classified_mask_lab[y,
                                x] = 255 if dist_to_bg_lab > threshold else 0

    return classified_mask_rgb, classified_mask_hsv, classified_mask_lab


# Function to evaluate the generated mask against the ground truth mask
def evaluate_mask(generated_mask, ground_truth_mask):
    # Compute Intersection over Union (IoU)
    intersection = np.logical_and(
        generated_mask == 255, ground_truth_mask == 255).sum()
    union = np.logical_or(generated_mask == 255,
                          ground_truth_mask == 255).sum()
    iou = intersection / union if union != 0 else 0
    return iou


def apply_mask_to_image(image, mask):
    # Make a copy of the original image
    masked_image = image.copy()
    # Apply mask: Set background pixels (where mask == 0) to black
    masked_image[mask == 0] = [0, 0, 0]

    return masked_image


def calculate_precision_recall_f1(predicted_mask, ground_truth_mask):
    # Convert masks to boolean arrays where True means foreground
    predicted_foreground = predicted_mask == 255
    ground_truth_foreground = ground_truth_mask == 255

    # True Positives (TP): Pixels correctly classified as foreground
    TP = np.logical_and(predicted_foreground, ground_truth_foreground).sum()

    # False Positives (FP): Pixels incorrectly classified as foreground
    FP = np.logical_and(predicted_foreground, np.logical_not(
        ground_truth_foreground)).sum()

    # False Negatives (FN): Pixels incorrectly classified as background
    FN = np.logical_and(np.logical_not(predicted_foreground),
                        ground_truth_foreground).sum()

    # Calculate Precision, Recall, and F1-score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision +
                                           recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def remove_background(image, classified_mask_rgb):
    # Ensure the mask is in the same size as the image and has 3 channels
    if len(classified_mask_rgb.shape) == 2:
        classified_mask_rgb = cv2.cvtColor(
            classified_mask_rgb, cv2.COLOR_GRAY2BGR)

    # Apply the mask to the image
    # Foreground will retain its colors, background will be black
    masked_image = cv2.bitwise_and(image, classified_mask_rgb)

    return masked_image


# Process all images in the folder and accumulate IoU performance
def process_folder_and_evaluate(image_folder):
    total_iou_rgb = 0
    total_iou_hsv = 0
    total_iou_lab = 0

    total_iou_rgb_closing = 0
    total_iou_hsv_closing = 0
    total_iou_lab_closing = 0

    num_images = 0
    kernel_size = 3
    # Iterate over all .jpg images in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            image_jpg_path = os.path.join(image_folder, filename)
            image_png_path = os.path.join(
                image_folder, filename.replace('.jpg', '.png'))

            # Load the image and the corresponding mask
            image_jpg = cv2.imread(image_jpg_path)
            image_png = cv2.imread(image_png_path, cv2.IMREAD_GRAYSCALE)

            if image_jpg is None or image_png is None:
                print(f"Error loading {filename}, skipping.")
                continue

            # Create background color models for RGB, HSV, and LAB
            avg_color_bg_rgb = create_background_model(image_jpg, bg_value=50)
            avg_color_bg_hsv = create_background_model(
                cv2.cvtColor(image_jpg, cv2.COLOR_BGR2HSV), bg_value=50)
            avg_color_bg_lab = create_background_model(
                cv2.cvtColor(image_jpg, cv2.COLOR_BGR2LAB), bg_value=50)

            # Classify the image using RGB, HSV, and LAB color spaces
            classified_mask_rgb, classified_mask_hsv, classified_mask_lab = classify_in_multiple_color_spaces(
                image_jpg, avg_color_bg_rgb, avg_color_bg_hsv, avg_color_bg_lab
            )

            # Evaluate the classified masks against the ground truth mask
            iou_rgb = evaluate_mask(classified_mask_rgb, image_png)
            iou_hsv = evaluate_mask(classified_mask_hsv, image_png)
            iou_lab = evaluate_mask(classified_mask_lab, image_png)

            # kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # opening = cv2.morphologyEx(
            #     classified_mask_rgb, cv2.MORPH_OPEN, kernel)
            # closing_rgb = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

            # opening = cv2.morphologyEx(
            #     classified_mask_hsv, cv2.MORPH_OPEN, kernel)
            # closing_hsv = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

            # opening = cv2.morphologyEx(
            #     classified_mask_lab, cv2.MORPH_OPEN, kernel)
            # closing_lab = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

            # iou_rgb_closing = evaluate_mask(closing_rgb, image_png)
            # iou_hsv_closing = evaluate_mask(closing_hsv, image_png)
            # iou_lab_closing = evaluate_mask(closing_lab, image_png)

            # Print evaluation results for the image
            print(f"Results for {filename}:")
            print(f"  IoU (RGB): {iou_rgb:.4f}")
            print(f"  IoU (HSV): {iou_hsv:.4f}")
            print(f"  IoU (LAB): {iou_lab:.4f}")

            # Accumulate the IoU for overall performance
            total_iou_rgb += iou_rgb
            total_iou_hsv += iou_hsv
            total_iou_lab += iou_lab

            # total_iou_rgb_closing += iou_rgb_closing
            # total_iou_hsv_closing += iou_hsv_closing
            # total_iou_lab_closing += iou_lab_closing
            num_images += 1

            masked_image_rgb = apply_mask_to_image(
                image_jpg, classified_mask_rgb)
            masked_image_hsv = apply_mask_to_image(
                image_jpg, classified_mask_hsv)
            masked_image_lab = apply_mask_to_image(
                image_jpg, classified_mask_lab)

            pcf = calculate_precision_recall_f1(classified_mask_rgb, image_png)
            print(
                f"precision = {pcf[0]}, recall = {pcf[1]}, f1_score = {pcf[2]}")

            print()

            masked_image_rgb = remove_background(
                image_jpg, classified_mask_rgb)
            masked_image_hsv = remove_background(
                image_jpg, classified_mask_hsv)
            masked_image_lab = remove_background(
                image_jpg, classified_mask_lab)

            # Save the classified masks
            cv2.imwrite(os.path.join(
                'Team2/datasets/masked_img', f'masked_{filename[:-4]}_rgb.png'), masked_image_rgb)
            cv2.imwrite(os.path.join(
                'Team2/datasets/masked_img', f'masked_{filename[:-4]}_hsv.png'), masked_image_hsv)
            cv2.imwrite(os.path.join(
                'Team2/datasets/masked_img', f'masked_{filename[:-4]}_lab.png'), masked_image_lab)

            # cv2.imshow('Masked Image', masked_image)
            # cv2.waitKey(0)

            dataset_folder = 'Team2/datasets/qsd2_w1'
            most_similar_image_rgb, similarity_score_rgb = find_most_similar_image(
                masked_image_rgb, dataset_folder)

            most_similar_image_hsv, similarity_score_hsv = find_most_similar_image(
                masked_image_hsv, dataset_folder)

            most_similar_image_lab, similarity_score_lab = find_most_similar_image(
                masked_image_lab, dataset_folder)

            print(f"Most similar image for rgb: {most_similar_image_rgb}")
            print(f"Similarity score for rgb: {similarity_score_rgb}")

            print(f"Most similar image for hsv: {most_similar_image_hsv}")
            print(f"Similarity score for hsv: {similarity_score_hsv}")

            print(f"Most similar image for lab: {most_similar_image_lab}")
            print(f"Similarity score for lab: {similarity_score_lab}")

            print("_____________________________________________________")

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
image_folder = 'Team2/datasets/qsd2_w1'
process_folder_and_evaluate(image_folder)


sys.stdout = old_stdout
log_file.close()
