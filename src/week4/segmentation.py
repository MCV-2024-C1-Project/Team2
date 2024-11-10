import cv2
import numpy as np
import os
#from skimage.segmentation import clear_border, chan_vese
from ColorSegmentation import chan_vese_segmentation_colors

print("\n")
print("Week 4 - Segmentation")
# Load the image
directory = 'filtered_qsd1_w4_adptv/'

# Global Parameters
# Changeable parameters
mu = 0.2
nu = 0
lambda1 = 1
lambda2 = 1

# Fixed parameters
epsilon = 1
time_step = 1e-2
eta = 1e-8
iterations = 1000
tolerance = 0.1

def check_and_reverse_border(mask, x = 5, threshold = 0.5):
    """
    Checks the borders of a binary mask with width `x`. If the percentage of white pixels
    in the borders exceeds the specified threshold, the function reverses the mask.
    
    Parameters:
    - mask: Binary mask (2D numpy array).
    - x: Width of the border to check.
    - threshold: Relative threshold for white pixels in the border (0 to 1).
    
    Returns:
    - mask: The (possibly inverted) binary mask.
    """
    # Get mask dimensions
    #height, width = mask.shape

    # Extract borders
    top_border = mask[:x, :]
    bottom_border = mask[-x:, :]
    left_border = mask[:, :x]
    right_border = mask[:, -x:]

    # Count white pixels in the borders
    total_border_pixels = top_border.size + bottom_border.size + left_border.size + right_border.size
    white_border_pixels = (
        np.sum(top_border == 255) +
        np.sum(bottom_border == 255) +
        np.sum(left_border == 255) +
        np.sum(right_border == 255)
    )

    # Calculate the percentage of white pixels in the border
    white_percentage = white_border_pixels / total_border_pixels

    # Reverse the mask if the white percentage exceeds the threshold
    if white_percentage > threshold:
        mask = cv2.bitwise_not(mask)
    
    return mask

def create_mask(directory, mu, nu, lambda1, lambda2):
# Iterate through the directory to process all .jpg files
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'): # Change string according to dataset
            img_path = os.path.join(directory, filename)
            print(f"Processing {img_path}...")

            # Load the image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype('float')
            
            if img is None:
                print(f"Error: {img_path} not found.")
                continue

            # The comment section might yield better resutls, check at later stage
            """img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_s = img_hsv[:,:,1]
            claeh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_s_clahe = claeh.apply(img_s)"""

            # Perform the Chan-Vese segmentation
            segmented_img = chan_vese_segmentation_colors(
                img, mu, nu, lambda1, lambda2, epsilon, time_step, iterations, eta, tolerance
            )

            # Check borders and reverse if necessary
            segmented_img = check_and_reverse_border(segmented_img, 10, 0.5)

            # Save the segmented image
            save_path = os.path.join(directory, filename.replace('.jpg', '.png'))
            cv2.imwrite(save_path, segmented_img)
            print(f"Saved {save_path}.")
        else:
            continue


create_mask(directory, mu, nu, lambda1, lambda2)

print('Finish the data folder proccessing')
