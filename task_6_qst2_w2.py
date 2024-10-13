import cv2
import numpy as np
import os
import pandas as pd
import pickle
import utils
import re


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
            # Load the image and the corresponding mask
            image_jpg = cv2.imread(image_jpg_path)
            if image_jpg is None:
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


def extract_number_from_filename(filename):
    '''Function to extract the number of the image'''
    match = re.search(r'bbdd_(\d{5})_w2\.pkl', filename)
    if match:
        return int(match.group(1))


def extract_number_from_filename_qsd1_w1(filename):
    '''Function to extract the number of the image'''
    match = re.search(r'(\d{5})_w2\.pkl', filename)
    if match:
        return int(match.group(1))


def spatial_pyramid_histogram(image, levels=2,resize=True,dimensions=1,hist_size=[8,8], hist_range=[0,256,0,256]):
    """
    Compute a spatial pyramid representation of histograms, with concatenation of histograms per channel.
    Level zero has 1 block. 2^0=1 so blocks 1*1=1
    Level one has 4 blocks. 2^1=2 so blocks 2*2=4
    Level two has 16 blocks. 2^2=4 so blocks 4*4=16
    """

    pyramid_hist = []
    if resize == True:
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]  # Get the height and width of the image

    # Loop through each level in the pyramid
    for level in range(levels + 1):
        num_blocks = 2 ** level  
        block_h, block_w = h // num_blocks, w // num_blocks  # Block size

        for i in range(num_blocks):
            for j in range(num_blocks):
                # Define the block region
                block = image[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
                # Compute histograms depending on the number of channels
                block_hist = []
                if dimensions == 1:
                    # Compute 1D histogram
                    for channel in range(3):
                        hist = cv2.calcHist([block], [channel], None, hist_size, hist_range)
                        hist /= hist.sum()  
                        hist = hist.flatten()  
                        block_hist.append(hist)

                elif dimensions == 2:
                    # Compute 2D histogram 
                    lab_hist2D = cv2.calcHist([block], [0, 1], None, hist_size, hist_range)
                    # Normalize the histogram (to the range [0, 1] with NORM_MINMAX)
                    normalized = cv2.normalize(lab_hist2D, lab_hist2D, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    # Flatten the 2D histogram into a 1D vector
                    flattened_hist = normalized.flatten()
                    block_hist.append(flattened_hist)

                elif dimensions == 3:
                    # Compute 3D histogram 
                    lab_hist3D = cv2.calcHist([block], [0, 1, 2], None, hist_size, hist_range)
                    # Normalize the histogram (to the range [0, 1] with NORM_MINMAX)
                    normalized = cv2.normalize(lab_hist3D, lab_hist3D, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    # Flatten the 3D histogram into a 1D vector
                    flattened_hist = normalized.flatten()
                    block_hist.append(flattened_hist)

                # Concatenate histograms for this block
                block_hist = np.concatenate(block_hist)
                pyramid_hist.append(block_hist)

    # Concatenate all block histograms into a single feature vector
    pyramid_hist = np.concatenate(pyramid_hist)
    return pyramid_hist


def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.png'):
            img_path = os.path.join(directory_path, filename)
            img_BGR = cv2.imread(img_path)

            # CieLab
            img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
            hist_resize_HSV_64_1D = spatial_pyramid_histogram(img_HSV, levels=2,resize=True, dimensions=1, hist_size=[64], hist_range=[0, 256])

            histograms = {
                'hist_resize_HSV_64_1D': hist_resize_HSV_64_1D,
            }

            save_path = directory_path
            pkl_filename = os.path.splitext(filename)[0] + '_w2.pkl'
            pkl_path = os.path.join(save_path, pkl_filename)
            print(pkl_path)
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(histograms, pkl_file)


# Define folder containing the images and the output folder
image_folder = 'datasets/qst2_w2'
output_folder = 'image_without_background_test'
mask_path = 'results/week2/QST2/method1'
# Process the folder and save the results in the new folder
process_folder_and_evaluate(image_folder, output_folder, mask_path)


# process both folders
directory_test1 = "image_without_background_test"
print("Current working directory:", os.getcwd())
print("Processing directory 1:")
process_directory(directory_test1)


directory = 'image_without_background_test'
directory_bbdd = 'data/BBDD/week2'


# Method 1
min_distance = float('inf')
list_results_k_10 = []

files = os.listdir(directory)
files_sorted = sorted(files)

for file_compare_image in files_sorted:
    # catch the first image then the second and so on of the qsd1_w1

    if file_compare_image.endswith('_w2.pkl') and file_compare_image != 'gt_corresps.pkl':
        pkl_grey_path = os.path.join(directory, file_compare_image)
        with open(pkl_grey_path, 'rb') as pkl_file:
            histograms_first = pickle.load(pkl_file)

        distances = []
        index_qsd1_w1 = extract_number_from_filename_qsd1_w1(file_compare_image)

        for filename in os.listdir(directory_bbdd):
            if filename.endswith('_w2.pkl') and filename != 'relationships.pkl':
                pkl_path = os.path.join(directory_bbdd, filename)

                with open(pkl_path, 'rb') as pkl_file:
                    histograms = pickle.load(pkl_file)

                histogram_first_grey = histograms_first['hist_resize_HSV_64_1D']
                histogram_grey = histograms['hist_resize_HSV_64_1D']
                # # Try all 4 loss functions: euc_dist, L1_dist, X2_distance, hellinger_kernel, histogram_similiarity
                distance = utils.our_metric(histogram_first_grey, histogram_grey)

                index = extract_number_from_filename(filename)

                distances.append((distance, index))

        # Sort the distances and select the top k results
        distances.sort(key=lambda x: x[0])  # Sort by distance (lowest first)
        top_k_10_results = [index for _, index in distances[:10]]

        # when the loop for the first query image is finish we save the index in a list
        list_results_k_10.append(top_k_10_results)

list_results_k_10 = [result for result in list_results_k_10]

# Para los resultados aplanados
predicted_flattened_k_10 = [p for p in list_results_k_10]

# To ensure that they are integers
list_results_k_10_enteros = [[int(x) for x in sublist] for sublist in list_results_k_10]

print(list_results_k_10_enteros)

# store results k=10
with open('results/week2/QST2/method1/result.pkl', 'wb') as pkl_file:
    pickle.dump(list_results_k_10_enteros, pkl_file)
