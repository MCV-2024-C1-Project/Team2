import cv2
import numpy as np


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


def contour_detection_mask(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a blank mask
    mask = np.zeros_like(gray)
    # Filter contours based on area (you can adjust the threshold)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjust the contour area threshold as needed
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    return mask


def background_subtraction_mask(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Create background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    # Apply the background subtractor
    fg_mask = backSub.apply(image)
    # Refine mask using morphological operations
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # Threshold the mask to ensure binary output
    _, binary_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

    return binary_mask


def hough_line_detection_mask(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 25, 50, apertureSize=3)
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    # Create a blank mask
    mask = np.zeros_like(gray)
    # Draw the lines on the mask
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)  # Line thickness of 3

    # Invert the mask to make the lines white
    mask = cv2.bitwise_not(mask)

    return mask


def adaptive_thresholding(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Convert to grayscale
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

    # Create a copy of the original image for drawing
    output_image = image.copy()

    # Draw each rectangle in descending order, labeling them with numbers
    for i, (x, y, w, h) in enumerate(rectangles):
        # Draw the rectangle on the output image
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Put the label of the rectangle number
        cv2.putText(output_image, f'#{i+1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Rectangles', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if i == 10:
            break

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


image_path = 'datasets/qsd2_w3/00022.jpg'

# Adaptive Thresholding
adaptive_mask, contour_mask, contours = adaptive_thresholding(image_path)
cv2.imshow('Adaptive Thresholding', adaptive_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Adaptive Thresholding', contour_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# exit here
exit()

# Edge Detection Mask
edge_mask, contour_mask, largest_rectangles = edge_detection_mask(image_path)
cv2.imshow('Edge Detection Mask', edge_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Edge Detection Contour Mask', contour_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.destroyAllWindows()

# exit here
exit()

# Contour Detection Mask
contour_mask = contour_detection_mask(image_path)
cv2.imshow('Contour Detection Mask', contour_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Background Subtraction Mask
bg_mask = background_subtraction_mask(image_path)
cv2.imshow('Background Subtraction Mask', bg_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Hough Line Detection Mask
hough_mask = hough_line_detection_mask(image_path)
cv2.imshow('Hough Line Detection Mask', hough_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
