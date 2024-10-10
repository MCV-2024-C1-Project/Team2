import os
import cv2
import numpy as np


# Function to determine the number of pixels we can move inward from each edge while encountering only background
def calculate_background_depth(mask):
    height, width = mask.shape

    # Initialize the distance from each edge to still be background
    top_distance = 0
    bottom_distance = 0
    left_distance = 0
    right_distance = 0

    # Calculate how far we can move in from the top edge
    for i in range(height):
        if np.all(mask[i, :] == 0):  # Check if the whole row is background
            top_distance += 1
        else:
            break

    # Calculate how far we can move in from the bottom edge
    for i in range(height - 1, -1, -1):
        if np.all(mask[i, :] == 0):  # Check if the whole row is background
            bottom_distance += 1
        else:
            break

    # Calculate how far we can move in from the left edge
    for i in range(width):
        if np.all(mask[:, i] == 0):  # Check if the whole column is background
            left_distance += 1
        else:
            break

    # Calculate how far we can move in from the right edge
    for i in range(width - 1, -1, -1):
        if np.all(mask[:, i] == 0):  # Check if the whole column is background
            right_distance += 1
        else:
            break

    return top_distance, bottom_distance, left_distance, right_distance


# Function to process all masks in a folder and find the minimum distance for all images
def find_min_background_depth(folder_path):
    # Get a list of all image files in the folder
    mask_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # Initialize minimum distances with a large value
    min_top = float('inf')
    min_bottom = float('inf')
    min_left = float('inf')
    min_right = float('inf')

    for mask_file in mask_files:
        mask_path = os.path.join(folder_path, mask_file)

        # Load the mask image in grayscale (binary)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Calculate how many pixels we can move inward from each edge and still be background
        top_distance, bottom_distance, left_distance, right_distance = calculate_background_depth(mask)

        # Update the minimum values for each edge
        min_top = min(min_top, top_distance)
        min_bottom = min(min_bottom, bottom_distance)
        min_left = min(min_left, left_distance)
        min_right = min(min_right, right_distance)

    # Output the minimum distances we can move inward for all masks
    print("Minimum distances you can move inward and still be background across all masks:")
    print(f"Top: {min_top} pixels")
    print(f"Bottom: {min_bottom} pixels")
    print(f"Left: {min_left} pixels")
    print(f"Right: {min_right} pixels")


# Example usage
folder_path = 'datasets/qsd2_w1'
find_min_background_depth(folder_path)
