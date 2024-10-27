import cv2
import numpy as np
import os
import pickle   
from skimage.segmentation import clear_border, chan_vese
from task1 import is_noisy, apply_filters

# Load the image
directory = 'datasets/qst2_w3/'

def check_and_reverse_border(mask, x, threshold):
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
    height, width = mask.shape

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

# Iterate through the directory to process all .jpg files
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        img_path = os.path.join(directory, filename)

        # Load the image
        img = cv2.imread(img_path)
        #img_denoised = apply_filters(img)
        img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_s = img_hsv[:,:,1]
        claeh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_s_clahe = claeh.apply(img_s)

        # Perform Chan-Vese segmentation
        cv = chan_vese(img_s_clahe, mu=0.08, lambda1=1, lambda2=1, tol=1e-3, max_num_iter=300,
                       dt=0.5, init_level_set="checkerboard", extended_output=True)

        # Convert the result to uint8 to display it
        cv_uint8 = (cv[0] * 255).astype(np.uint8)

        new_mask = check_and_reverse_border(cv_uint8, x=5, threshold=0.4)

        # Clear the border of the segmented image (cv_uint8)
        mask = clear_border(cv_uint8)

        # Store masks in a dictionary
        masks = {
            'mask': new_mask,
            'mask_clear_border': mask
        }

        # Save the masks as a .pkl file
        #pkl_filename = os.path.splitext(filename)[0] + '_seg.pkl'
        #pkl_path = os.path.join(directory, pkl_filename)
        
        #with open(pkl_path, 'wb') as pkl_file:
            #pickle.dump(masks, pkl_file)

        # Save both masks as PNG files
        mask_png_filename = os.path.splitext(filename)[0] + '_mask_s.png'
        mask_clear_border_png_filename = os.path.splitext(filename)[0] + '_mask_clear_border_s.png'

        mask_png_path = os.path.join(directory, mask_png_filename)
        mask_clear_border_png_path = os.path.join(directory, mask_clear_border_png_filename)

        # Save the mask images
        cv2.imwrite(mask_png_path, new_mask)
        cv2.imwrite(mask_clear_border_png_path, mask)

        print(f"Saved {mask_png_filename} and {mask_clear_border_png_filename}.")
            
print('Finish the data folder proccessing')
        
