import os
import numpy as np
import cv2


def process_and_save_masks(pkl_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all mask files in the folder
    for filename in os.listdir(pkl_folder):
        if filename.endswith('_mask_s_contour1.png'):
            contour2_filename = filename.replace('_mask_s_contour1.png', '_mask_s_contour2.png')

            if contour2_filename in os.listdir(pkl_folder):
                # Load two masks and join them
                image_path1 = os.path.join(pkl_folder, filename)
                image_path2 = os.path.join(pkl_folder, contour2_filename)
                mask1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
                mask2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
                joined_mask = np.maximum(mask1, mask2)

                # Save the single mask
                # output_filename = filename.replace('_mask_s_contour1.png', '_mask.png')

                # Save the single mask in results
                output_filename = filename.replace('_mask_s_contour1.png', '.png')

                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, joined_mask)
                print(f"Saved joined mask to {output_path}")
            else:
                # Load the single mask
                image_path = os.path.join(pkl_folder, filename)
                mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Save the single mask
                # output_filename = filename.replace('_mask_s_contour1.png', '_mask.png')

                # Save the single mask in results
                output_filename = filename.replace('_mask_s_contour1.png', '.png')

                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, mask)
                print(f"Saved single mask to {output_path}")


# Specify the input and output directories
input_directory = '../../datasets/qst2_w3'
# output_directory = '../../datasets/qst2_w3'
output_directory = '../../results/week3/QST2/method1'

# Process and save the masks
process_and_save_masks(input_directory, output_directory)
