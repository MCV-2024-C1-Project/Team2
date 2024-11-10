import cv2
import numpy as np
import os


def edge_detection_mask(image_path, output_folder):
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
        is_overlapping = False
        for rect2 in largest_rectangles:
            x2, y2, w2, h2 = rect2
            if not (x1 + w1 < x2 or x1 > x2 + w2 or y1 + h1 < y2 or y1 > y2 + h2):
                is_overlapping = True
                break
        if not is_overlapping:
            largest_rectangles.append(rect1)
        if len(largest_rectangles) == 2:
            break

    if len(largest_rectangles) == 2:
        area1 = largest_rectangles[0][2] * largest_rectangles[0][3]
        area2 = largest_rectangles[1][2] * largest_rectangles[1][3]
        if area2 < 0.2 * area1:
            largest_rectangles = [largest_rectangles[0]]

    contour_masks = []
    for i, rect in enumerate(largest_rectangles):
        contour_mask = np.zeros_like(mask)
        x, y, w, h = rect
        cv2.rectangle(contour_mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)
        contour_masks.append(contour_mask)

        # Save the contour mask
        output_path = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_edge_contour{i+1}.png")
        cv2.imwrite(output_path, contour_mask)

    return mask, contour_masks


def adaptive_thresholding(image_path, output_folder):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        2
    )
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(adaptive_mask, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [cv2.boundingRect(contour) for contour in contours]
    rectangles = sorted(rectangles, key=lambda r: r[2] * r[3], reverse=True)

    largest_rectangles = []
    for i, rect1 in enumerate(rectangles):
        x1, y1, w1, h1 = rect1
        is_overlapping = False
        for rect2 in largest_rectangles:
            x2, y2, w2, h2 = rect2
            if not (x1 + w1 < x2 or x1 > x2 + w2 or y1 + h1 < y2 or y1 > y2 + h2):
                is_overlapping = True
                break
        if not is_overlapping:
            largest_rectangles.append(rect1)
        if len(largest_rectangles) == 2:
            break

    if len(largest_rectangles) == 2:
        area1 = largest_rectangles[0][2] * largest_rectangles[0][3]
        area2 = largest_rectangles[1][2] * largest_rectangles[1][3]
        if area2 < 0.2 * area1:
            largest_rectangles = [largest_rectangles[0]]

    contour_masks = []
    for i, rect in enumerate(largest_rectangles):
        contour_mask = np.zeros_like(adaptive_mask)
        x, y, w, h = rect
        cv2.rectangle(contour_mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)
        contour_masks.append(contour_mask)

        # Save the contour mask
        output_path = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_adptv_contour{i+1}.png")
        cv2.imwrite(output_path, contour_mask)

    return adaptive_mask, contour_masks


def process_folder_and_evaluate(image_folder, output_folder):
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            image_jpg_path = os.path.join(image_folder, filename)

            adaptive_mask, adaptive_contours = adaptive_thresholding(image_jpg_path, output_folder)
            # edge_mask, edge_contours = edge_detection_mask(image_jpg_path, output_folder)

            # Save main masks
            cv2.imwrite(os.path.join(output_folder, f"{filename.split('.')[0]}_adptv.png"), adaptive_mask)
            # cv2.imwrite(os.path.join(output_folder, f"{filename.split('.')[0]}_edge.png"), edge_mask)


# Define input and output folder paths
image_folder = 'filtered_qsd1_w4_adptv'
output_folder = 'filtered_qsd1_w4_adptv'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Run the evaluation
process_folder_and_evaluate(image_folder, output_folder)
