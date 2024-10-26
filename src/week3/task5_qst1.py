import os
import sys
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct
import pickle
import re
import utils


def is_noisy(image):

    channels = cv2.split(image)  # This will give three channels: B, G, R

    laplacian_vars = []

    # Loop over each channel (B, G, R) and compute the Laplacian variance
    for channel in channels:
        laplacian = cv2.Laplacian(channel, cv2.CV_64F)
        laplacian_vars.append(laplacian.var())

    total_var = np.mean(laplacian_vars)

    return total_var > 4000


def compute_brightness(image):

    B, G, R = cv2.split(image)

    # Compute the brightness using the weighted sum
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    # brightness = 0.2126 * R + 0.7152 * G + 0.0722 * B

    # Compute the average brightness
    average_brightness = np.mean(brightness)

    return average_brightness


def get_image_index(filename):

    base_name = filename.replace('.jpg', '')
    index = int(base_name)

    return index


def multiply_hue(image, hue_factor):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    # Multiply the hue channel by the factor
    h = np.uint8((h.astype(np.float32) * hue_factor) %
                 180)  # Hue values wrap around at 180

    # Merge the channels back
    hsv_modified = cv2.merge([h, s, v])

    transformed_image = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)

    return transformed_image


def apply_filters(image):
    if is_noisy(image):
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
    else:
        denoised = cv2.medianBlur(image, 5)  # Less aggressive filtering

    return denoised


def calculate_psnr(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    if mse == 0:
        return 100

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def lbp_descriptor_by_block_size(image, num_points=8, radius=1, block_size=8):
    fixed_size = (256, 256)
    resized_image = cv2.resize(image, fixed_size)
    h, w = resized_image.shape[:2]
    lbp_blocks = []

    # Loop over the blocks defined by block_size
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Define the block region
            block = resized_image[i:i + block_size, j:j + block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size: 
                if len(image.shape) == 2:  # Grayscale image
                    # Apply LBP to the block
                    lbp = local_binary_pattern(block, num_points, radius, method="uniform")
                    lbp_uint8 = np.uint8(lbp)
                    # Compute histogram of the LBP result
                    hist, _ = np.histogram(lbp_uint8.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
                    hist = hist / np.sum(hist)
                    lbp_blocks.append(hist)  # Store the histogram for the block
                else:  # Color image (RGB)
                    for channel in range(3):
                        # Apply LBP to the corresponding channel
                        lbp = local_binary_pattern(block[:, :, channel], num_points, radius, method="uniform")
                        lbp_uint8 = np.uint8(lbp)
                        # Compute histogram of the LBP result
                        hist, _ = np.histogram(lbp_uint8.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
                        hist = hist / np.sum(hist)
                        lbp_blocks.append(hist)  

    # Concatenate all block histograms into a single feature vector
    feature_vector = np.concatenate(lbp_blocks)
    return feature_vector 

def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(directory_path, filename)
            img_BGR = cv2.imread(img_path)
            #CieLab
            img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
            hist_LBP_LAB_n8_r2_1D=lbp_descriptor_by_block_size(img_LAB,num_points=8,radius=2,block_size=8)

            histograms = {
                'hist_LBP_LAB_n8_r2_1D':hist_LBP_LAB_n8_r2_1D,
            }

            pkl_filename = os.path.splitext(filename)[0] + '_w3.pkl'
            pkl_path = os.path.join(directory_path, pkl_filename)
            print(pkl_path)
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(histograms, pkl_file)

def extract_number_from_filename(filename):
    '''Function to extract the number of the image'''
    match = re.search(r'bbdd_(\d{5})_w3\.pkl', filename)
    if match:
        return int(match.group(1))

def extract_number_from_filename_qsd1_w1(filename):
    '''Function to extract the number of the image'''
    match = re.search(r'(\d{5})_w3\.pkl', filename)
    if match:
        return int(match.group(1))

if __name__ == "__main__":
    image_path = '../../datasets/qst1_w3/'
    hue_threshold = 100
    for filename in os.listdir(image_path):
        if filename.lower().endswith(('.jpg')):

            image = cv2.imread(image_path + filename)
            # img_index = get_image_index(filename)

            average_brightness = compute_brightness(image)

            if is_noisy(image):

                best_result = None
                best_psnr = 0

                for h in range(10, 30, 5):
                    filtered_image = cv2.fastNlMeansDenoisingColored(
                        image, None, h, 10, 7, 21)
                    psnr = calculate_psnr(image, filtered_image)

                    if psnr > best_psnr:
                        best_psnr = psnr
                        best_result = filtered_image

                        kernel = np.ones((3, 3), np.uint8)
                        best_result = cv2.morphologyEx(
                            filtered_image, cv2.MORPH_OPEN, kernel)

                cv2.imwrite('filtered_test_img/' + filename, best_result)

            elif average_brightness > hue_threshold:

                hue_factor = 1.5
                transformed_image = multiply_hue(image, hue_factor)

                cv2.imwrite('filtered_test_img/' + filename, transformed_image)

            else:
                cv2.imwrite('filtered_test_img/' + filename, image)

    # #Process the descriptors
    directory_test = "filtered_test_img"
    print("Processing directory test:")
    process_directory(directory_test)

    #Result part
    directory_bbdd = '../../data/BBDD/week3'
    min_distance = float('inf')
    list_results_k_10 = []

    files = os.listdir(directory_test)
    files_sorted = sorted(files)

    for file_compare_image in files_sorted:
        # catch the first image then the second and so on of the qsd1_w1

        if file_compare_image.endswith('_w3.pkl') and file_compare_image != 'gt_corresps.pkl':
            pkl_grey_path = os.path.join(directory_test, file_compare_image)
            with open(pkl_grey_path, 'rb') as pkl_file:
                histograms_first = pickle.load(pkl_file)

            distances = []
            index_qsd1_w1 = extract_number_from_filename_qsd1_w1(file_compare_image)

            for filename in os.listdir(directory_bbdd):
                if filename.endswith('_w3.pkl') and filename != 'relationships.pkl':
                    pkl_path = os.path.join(directory_bbdd, filename)

                    with open(pkl_path, 'rb') as pkl_file:
                        histograms = pickle.load(pkl_file)

                    histogram_first_grey = histograms_first['hist_LBP_LAB_n8_r2_1D']
                    histogram_grey = histograms['hist_LBP_LAB_n8_r2_1D']
                    # # Try all 4 loss functions: euc_dist, L1_dist, X2_distance, hellinger_kernel, histogram_similiarity
                    distance = utils.L1_dist(histogram_first_grey, histogram_grey)

                    index = extract_number_from_filename(filename)

                    distances.append((distance, index))

            # Sort the distances and select the top k results
            distances.sort(key=lambda x: x[0])  # Sort by distance (lowest first)
            top_k_10_results = [index for _, index in distances[:10]]

            # when the loop for the first query image is finish we save the index in a list
            list_results_k_10.append(top_k_10_results)

    list_results_k_10 = [result for result in list_results_k_10]
    predicted_flattened_k_10 = [p for p in list_results_k_10]
    list_results_k_10_enteros = [[int(x) for x in sublist] for sublist in list_results_k_10]

    print(list_results_k_10_enteros)

    # store results k=10
    with open('../../results/week3/QST1/method1/result.pkl', 'wb') as pkl_file:
        pickle.dump(list_results_k_10_enteros, pkl_file)