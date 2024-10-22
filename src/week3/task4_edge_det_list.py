import cv2
import numpy as np
import os


def edge_detection_mask(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1=25, threshold2=50)
    # Dilate edges to close gaps
    edges_dilated = cv2.dilate(edges, None, iterations=1)
    # Invert the image to make the foreground white
    mask = cv2.bitwise_not(edges_dilated)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding rectangles for all contours
    rectangles = [cv2.boundingRect(contour) for contour in contours]
    # Sort rectangles by area in descending order
    rectangles = sorted(rectangles, key=lambda r: r[2] * r[3], reverse=True)

    # Find the two largest non-overlapping rectangles
    largest_rectangles = []
    for i, rect1 in enumerate(rectangles):
        x1, y1, w1, h1 = rect1
        # Check if this rectangle overlaps with any already selected rectangles
        is_overlapping = False
        for rect2 in largest_rectangles:
            x2, y2, w2, h2 = rect2
            # Check for overlap
            if not (x1 + w1 < x2 or x1 > x2 + w2 or y1 + h1 < y2 or y1 > y2 + h2):
                is_overlapping = True
                break

        # If it's not overlapping, add it to the list
        if not is_overlapping:
            largest_rectangles.append(rect1)

        # Stop if we have found two non-overlapping rectangles
        if len(largest_rectangles) == 2:
            break

    # Additional check for relative size difference between the two largest rectangles
    if len(largest_rectangles) == 2:
        # Calculate the area of the largest and second-largest rectangles
        area1 = largest_rectangles[0][2] * largest_rectangles[0][3]
        area2 = largest_rectangles[1][2] * largest_rectangles[1][3]

        # If the second-largest rectangle is significantly smaller than the largest one, keep only the largest
        if area2 < 0.2 * area1:  # Adjust the threshold (0.2) as needed
            largest_rectangles = [largest_rectangles[0]]

    # Create a blank mask to draw the rectangles
    contour_mask = np.zeros_like(mask)

    # Draw the rectangles on the mask
    for rect in largest_rectangles:
        x, y, w, h = rect
        cv2.rectangle(contour_mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)

    return mask, contour_mask, largest_rectangles


def adaptive_thresholding(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert to grayscalecub200_R_Margin_b06_150.log
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding
    adaptive_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,  # Block size, must be an odd number (tune this as needed)
        2    # Constant subtracted from the mean or weighted mean (tune this as needed)
    )

    # Define a kernel size for morphological operations
    kernel = np.ones((5, 5), np.uint8)
    # Apply closing to fill small holes in the foreground
    refined_mask = cv2.morphologyEx(adaptive_mask, cv2.MORPH_CLOSE, kernel)
    # Apply opening to remove small noise
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding rectangles for all contours
    rectangles = [cv2.boundingRect(contour) for contour in contours]
    # Sort rectangles by area in descending order
    rectangles = sorted(rectangles, key=lambda r: r[2] * r[3], reverse=True)

    # Find the two largest non-overlapping rectangles
    largest_rectangles = []
    for i, rect1 in enumerate(rectangles):
        x1, y1, w1, h1 = rect1
        # Check if this rectangle overlaps with any already selected rectangles
        is_overlapping = False
        for rect2 in largest_rectangles:
            x2, y2, w2, h2 = rect2
            # Check for overlap
            if not (x1 + w1 < x2 or x1 > x2 + w2 or y1 + h1 < y2 or y1 > y2 + h2):
                is_overlapping = True
                break

        # If it's not overlapping, add it to the list
        if not is_overlapping:
            largest_rectangles.append(rect1)

        # Stop if we have found two non-overlapping rectangles
        if len(largest_rectangles) == 2:
            break

    # Additional check for relative size difference between the two largest rectangles
    if len(largest_rectangles) == 2:
        # Calculate the area of the largest and second-largest rectangles
        area1 = largest_rectangles[0][2] * largest_rectangles[0][3]
        area2 = largest_rectangles[1][2] * largest_rectangles[1][3]

        # If the second-largest rectangle is significantly smaller than the largest one, keep only the largest
        if area2 < 0.2 * area1:  # Adjust the threshold (0.2) as needed
            largest_rectangles = [largest_rectangles[0]]

    # Create a blank mask to draw the rectangles
    contour_mask = np.zeros_like(adaptive_mask)

    # Draw the rectangles on the mask
    for rect in largest_rectangles:
        x, y, w, h = rect
        cv2.rectangle(contour_mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)

    return adaptive_mask, contour_mask, largest_rectangles


# Function to calculate precision, recall, and F1-score
def evaluate_mask(generated_mask, ground_truth_mask):
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


# Process all images in the folder and evaluate metrics for both functions
def process_folder_and_evaluate(image_folder):
    total_precision_adaptive = 0
    total_recall_adaptive = 0
    total_f1_adaptive = 0

    total_precision_edge = 0
    total_recall_edge = 0
    total_f1_edge = 0

    num_images = 0

    # Iterate over all .jpg images in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            image_jpg_path = os.path.join(image_folder, filename)
            image_png_path = os.path.join(image_folder, filename.replace('.jpg', '.png'))

            # Load the image and the corresponding ground truth mask
            image_jpg = cv2.imread(image_jpg_path)
            ground_truth_mask = cv2.imread(image_png_path, cv2.IMREAD_GRAYSCALE)

            if image_jpg is None or ground_truth_mask is None:
                print(f"Error loading {filename}, skipping.")
                continue

            # Generate the mask using the adaptive_thresholding function
            _, adaptive_mask, _ = adaptive_thresholding(image_jpg_path)

            # Generate the mask using the edge_detection_mask function
            _, edge_mask, _ = edge_detection_mask(image_jpg_path)

            # Evaluate the generated masks against the ground truth mask using precision, recall, and F1-score
            precision_adaptive, recall_adaptive, f1_adaptive = evaluate_mask(adaptive_mask, ground_truth_mask)
            precision_edge, recall_edge, f1_edge = evaluate_mask(edge_mask, ground_truth_mask)

            # Print evaluation results for the image
            print(f"Results for {filename}:")
            print(f"  Adaptive Thresholding - Precision: {precision_adaptive:.4f}, Recall: {recall_adaptive:.4f}, F1-score: {f1_adaptive:.4f}")
            print(f"  Edge Detection       - Precision: {precision_edge:.4f}, Recall: {recall_edge:.4f}, F1-score: {f1_edge:.4f}")

            # Accumulate the metrics for overall performance (adaptive thresholding)
            total_precision_adaptive += precision_adaptive
            total_recall_adaptive += recall_adaptive
            total_f1_adaptive += f1_adaptive

            # Accumulate the metrics for overall performance (edge detection)
            total_precision_edge += precision_edge
            total_recall_edge += recall_edge
            total_f1_edge += f1_edge

            num_images += 1

    # Compute and print the average performance over all images
    if num_images > 0:
        avg_precision_adaptive = total_precision_adaptive / num_images
        avg_recall_adaptive = total_recall_adaptive / num_images
        avg_f1_adaptive = total_f1_adaptive / num_images

        avg_precision_edge = total_precision_edge / num_images
        avg_recall_edge = total_recall_edge / num_images
        avg_f1_edge = total_f1_edge / num_images

        print("\nOverall Performance:")
        print("Adaptive Thresholding:")
        print(f"  Average Precision: {avg_precision_adaptive:.4f}")
        print(f"  Average Recall: {avg_recall_adaptive:.4f}")
        print(f"  Average F1-score: {avg_f1_adaptive:.4f}")
        print("\nEdge Detection:")
        print(f"  Average Precision: {avg_precision_edge:.4f}")
        print(f"  Average Recall: {avg_recall_edge:.4f}")
        print(f"  Average F1-score: {avg_f1_edge:.4f}")
    else:
        print("No valid images were processed.")


# Folder path for the images
image_folder = 'datasets/qsd2_w3'

# Evaluate both methods on the specified folder
process_folder_and_evaluate(image_folder)
