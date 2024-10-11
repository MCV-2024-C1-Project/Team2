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
    right_strip = image[:, -bg_value:, :]       # Right bg_value columns (full width)

    # Compute the average color of the top-bottom and left-right regions separately
    avg_color_top_bottom = np.mean(np.vstack((top_strip, bottom_strip)), axis=(0, 1))
    avg_color_left_right = np.mean(np.hstack((left_strip, right_strip)), axis=(0, 1))

    # Combine the two averages to form the final background model
    avg_color_bg = (avg_color_top_bottom + avg_color_left_right) / 2

    return avg_color_bg


# Function to calculate precision, recall, and F1-score
def evaluate_mask_precision_recall_f1(generated_mask, ground_truth_mask):
    # True Positive (TP): Both ground truth and predicted are foreground
    TP = np.logical_and(generated_mask == 255, ground_truth_mask == 255).sum()

    # False Positive (FP): Predicted is foreground, but ground truth is background
    FP = np.logical_and(generated_mask == 255, ground_truth_mask == 0).sum()

    # False Negative (FN): Predicted is background, but ground truth is foreground
    FN = np.logical_and(generated_mask == 0, ground_truth_mask == 255).sum()

    # Precision: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Recall: TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1-score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


# Process all images in the folder and accumulate Precision, Recall, and F1-score
def process_folder_and_evaluate(image_folder):
    total_precision_hsv = 0
    total_recall_hsv = 0
    total_f1_hsv = 0
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

            # Create background color models for HSV
            avg_color_bg_hsv = create_background_model(cv2.cvtColor(image_jpg, cv2.COLOR_BGR2HSV), bg_value=50)

            # Classify the image using HSV color space
            classified_mask_hsv = np.zeros((image_jpg.shape[0], image_jpg.shape[1]), dtype=np.uint8)
            image_hsv = cv2.cvtColor(image_jpg, cv2.COLOR_BGR2HSV)

            for y in range(image_jpg.shape[0]):
                for x in range(image_jpg.shape[1]):
                    pixel_hsv = image_hsv[y, x]
                    dist_to_bg_hsv = np.linalg.norm(pixel_hsv - avg_color_bg_hsv)
                    classified_mask_hsv[y, x] = 255 if dist_to_bg_hsv > 50 else 0

            # Evaluate the classified masks against the ground truth mask using precision, recall, and F1-score
            precision_hsv, recall_hsv, f1_hsv = evaluate_mask_precision_recall_f1(classified_mask_hsv, image_png)

            # Print evaluation results for the image
            print(f"Results for {filename}:")
            print(f"  Precision (HSV): {precision_hsv:.4f}")
            print(f"  Recall (HSV): {recall_hsv:.4f}")
            print(f"  F1-score (HSV): {f1_hsv:.4f}")

            # Accumulate the metrics for overall performance
            total_precision_hsv += precision_hsv
            total_recall_hsv += recall_hsv
            total_f1_hsv += f1_hsv
            num_images += 1

    # Compute and print the average performance over all images
    avg_precision_hsv = total_precision_hsv / num_images
    avg_recall_hsv = total_recall_hsv / num_images
    avg_f1_hsv = total_f1_hsv / num_images

    print("\nOverall Performance:")
    print(f"  Average Precision (HSV): {avg_precision_hsv:.4f}")
    print(f"  Average Recall (HSV): {avg_recall_hsv:.4f}")
    print(f"  Average F1-score (HSV): {avg_f1_hsv:.4f}")


# Define folder containing the images
image_folder = 'datasets/qsd2_w1'
process_folder_and_evaluate(image_folder)
