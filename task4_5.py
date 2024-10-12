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


# Function to remove background from the image using the classified mask
def remove_background(image, mask):
    # Convert the mask to 3 channels to apply it to the original image
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Apply the mask to the image (bitwise_and keeps only the foreground pixels)
    image_without_bg = cv2.bitwise_and(image, mask_3channel)
    
    # Create an alpha channel (transparency mask)
    alpha_channel = np.where(mask == 255, 255, 0).astype(np.uint8)  # Foreground is opaque, background is transparent
    
    # Add the alpha channel to the image
    image_with_alpha = cv2.merge([image_without_bg[:, :, 0], image_without_bg[:, :, 1], image_without_bg[:, :, 2], alpha_channel])
    
    return image_with_alpha


# Function to clean up the mask by removing small black lines
def clean_mask(mask, threshold=0.8):
    # Iterate over each row (horizontal lines)
    for y in range(mask.shape[0]):
        # Calculate the percentage of black pixels (0)
        black_pixel_percentage = np.mean(mask[y] == 0)
        # If more than threshold percentage are black, set the entire row to black
        if black_pixel_percentage > threshold:
            mask[y] = 0

    # Iterate over each column (vertical lines)
    for x in range(mask.shape[1]):
        # Calculate the percentage of black pixels (0)
        black_pixel_percentage = np.mean(mask[:, x] == 0)
        # If more than threshold percentage are black, set the entire column to black
        if black_pixel_percentage > threshold:
            mask[:, x] = 0

    return mask

def morphologically_close_mask(mask):
    # Define a kernel for the morphological operations
    kernel = np.ones((5, 5), np.uint8)  # You can adjust the size of the kernel

    # Apply closing (dilate followed by erode) to fill small holes
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return closed_mask


# Process all images in the folder and accumulate Precision, Recall, and F1-score
def process_folder_and_evaluate(image_folder, output_folder, mask_path):
    total_precision_hsv = 0
    total_recall_hsv = 0
    total_f1_hsv = 0
    num_images = 0

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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

            # Create background color model for HSV
            avg_color_bg_hsv = create_background_model(cv2.cvtColor(image_jpg, cv2.COLOR_BGR2HSV), bg_value=50)

            # Classify the image using HSV color space
            classified_mask_hsv = np.zeros((image_jpg.shape[0], image_jpg.shape[1]), dtype=np.uint8)
            image_hsv = cv2.cvtColor(image_jpg, cv2.COLOR_BGR2HSV)

            for y in range(image_jpg.shape[0]):
                for x in range(image_jpg.shape[1]):
                    pixel_hsv = image_hsv[y, x]
                    dist_to_bg_hsv = np.linalg.norm(pixel_hsv - avg_color_bg_hsv)
                    classified_mask_hsv[y, x] = 255 if dist_to_bg_hsv > 50 else 0

            # Clean the mask to remove small black lines
            cleaned_mask = clean_mask(classified_mask_hsv, threshold=0.8)

            # Apply morphological closing to fill small holes in the mask
            closed_mask = morphologically_close_mask(cleaned_mask)

            # Remove the background from the image and add transparency
            image_without_background = remove_background(image_jpg, closed_mask)

            # Create the output path in the new folder
            output_path = os.path.join(output_folder, filename.replace('.jpg', '_without_bg.png'))
            # Save the image without background (in PNG format with transparency)
            cv2.imwrite(output_path, image_without_background)
            print(f"Saved image without background: {output_path}")

            closed_mask_path = os.path.join(mask_path, filename.replace('.jpg', '.png'))
            cv2.imwrite(closed_mask_path, closed_mask)
            print(f"Saved closed mask: {closed_mask_path}")

            # Evaluate the classified masks against the ground truth mask using precision, recall, and F1-score
            precision_hsv, recall_hsv, f1_hsv = evaluate_mask_precision_recall_f1(closed_mask, image_png)

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


# Define folder containing the images and the output folder
image_folder = 'datasets/qsd2_w2'
output_folder = 'image_without_background'
mask_path = 'results/week2/QST2/method1'

# Process the folder and save the results in the new folder
process_folder_and_evaluate(image_folder, output_folder, mask_path)
