import cv2
import numpy as np
import os
import pickle   
from skimage.segmentation import clear_border, chan_vese
from task1 import is_noisy, apply_filters

# Load the image
directory = 'datasets/qst2_w3/'

# Iterate through the directory to process all .jpg files
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        img_path = os.path.join(directory, filename)

        # Load the image
        img = cv2.imread(img_path)
        img_denoised = apply_filters(img)
        img_greyscale = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2HSV)
        img_s = img_hsv[:,:,1]
        # Perform Chan-Vese segmentation
        cv = chan_vese(img_s, mu=0.08, lambda1=1, lambda2=1, tol=1e-3, max_num_iter=200,
                       dt=0.5, init_level_set="checkerboard", extended_output=True)

        # Convert the result to uint8 to display it
        cv_uint8 = (cv[0] * 255).astype(np.uint8)

        # Clear the border of the segmented image (cv_uint8)
        mask = clear_border(cv_uint8)

        # Store masks in a dictionary
        masks = {
            'mask': cv_uint8,
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
        cv2.imwrite(mask_png_path, cv_uint8)
        cv2.imwrite(mask_clear_border_png_path, mask)

        print(f"Saved {mask_png_filename} and {mask_clear_border_png_filename}.")
            
print('Finish the data folder proccessing')
        
