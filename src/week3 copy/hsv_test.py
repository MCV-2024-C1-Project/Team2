import cv2
from skimage.segmentation import clear_border, chan_vese
import numpy as np

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

img = cv2.imread('datasets/qsd2_w3/00002.jpg')
 
cv2.imshow('image', img); cv2.waitKey(0); cv2.destroyAllWindows()

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('image', img_hsv); cv2.waitKey(0); cv2.destroyAllWindows()

img_h = img_hsv[:,:,0]

cv2.imshow('image', img_h); cv2.waitKey(0); cv2.destroyAllWindows()

img_s = img_hsv[:,:,1]

cv2.imshow('image', img_s); cv2.waitKey(0); cv2.destroyAllWindows()

claeh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_s_clahe = claeh.apply(img_s)

cv2.imshow('image', img_s_clahe); cv2.waitKey(0); cv2.destroyAllWindows()

cv = chan_vese(img_s_clahe, mu=0.08, lambda1=1, lambda2=1, tol=1e-3, max_num_iter=300,
                       dt=0.5, init_level_set="checkerboard", extended_output=True)

# Convert the result to uint8 to display it
cv_uint8 = (cv[0] * 255).astype(np.uint8)

cv2.imshow('image', cv_uint8); cv2.waitKey(0); cv2.destroyAllWindows()

new_mask = check_and_reverse_border(cv_uint8, x=5, threshold=0.4)

cv2.imshow('image', new_mask); cv2.waitKey(0); cv2.destroyAllWindows()

mask = clear_border(cv_uint8)

cv2.imshow('image', mask); cv2.waitKey(0); cv2.destroyAllWindows()