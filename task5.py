import re
import os
import pickle
import pandas as pd
from src.week2 import utils
import cv2
import numpy as np


def spatial_pyramid_histogram(image, levels=2,resize=False, dimensions=1,hist_size=[8,8], hist_range=[0,256,0,256]):
    """
    Compute a spatial pyramid representation of histograms, with concatenation of histograms per channel.
    Level zero has 1 block. 2^0=1 so blocks 1*1=1
    Level one has 4 blocks. 2^1=2 so blocks 2*2=4
    Level two has 16 blocks. 2^2=4 so blocks 4*4=16
    """

    pyramid_hist = []
    # resize image to 256*256
    if resize==True:
        image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA) 
    h, w = image.shape[:2]  # Get the height and width of the image

    # Loop through each level in the pyramid
    for level in range(levels + 1):
        num_blocks = 2 ** level  
        block_h, block_w = h // num_blocks, w // num_blocks  # Block size

        for i in range(num_blocks):
            for j in range(num_blocks):
                # Define the block region
                block = image[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
                #print(f'block ' + str(i) +' : '+ str(j))
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
           
            #CieLab
            img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
            hist_LAB_8_1D=spatial_pyramid_histogram(img_LAB, levels=2,resize=False, dimensions=1,hist_size=[8], hist_range=[0, 256])
            hist_LAB_32_2D = spatial_pyramid_histogram(img_LAB, levels=2,resize=False, dimensions=2, hist_size=[32,32], hist_range=[0, 256,0, 256])
            hist_resize_LAB_64_1D = spatial_pyramid_histogram(img_LAB, levels=2,resize=True, dimensions=1, hist_size=[64], hist_range=[0, 256])
            #HSV
            img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
            hist_HSV_8_1D=spatial_pyramid_histogram(img_HSV, levels=2,resize=False, dimensions=1,hist_size=[8], hist_range=[0, 256])
            hist_HSV_8_3D = spatial_pyramid_histogram(img_HSV, levels=2,resize=False, dimensions=3,hist_size=[8,8,8], hist_range=[0, 180,0, 256,0, 256])
            hist_resize_HSV_64_1D = spatial_pyramid_histogram(img_HSV, levels=2,resize=True, dimensions=1, hist_size=[64], hist_range=[0, 256])

            histograms = {
                'hist_LAB_8_1D':hist_LAB_8_1D,
                'hist_LAB_32_2D': hist_LAB_32_2D,
                'hist_HSV_8_1D':hist_HSV_8_1D,
                'hist_HSV_8_3D': hist_HSV_8_3D,
                'hist_resize_LAB_64_1D': hist_resize_LAB_64_1D,
                'hist_resize_HSV_64_1D': hist_resize_HSV_64_1D

            }

            save_path = directory_path 
            pkl_filename = os.path.splitext(filename)[0] + '_w2.pkl'
            pkl_path = os.path.join(save_path, pkl_filename)
            print(pkl_path)
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(histograms, pkl_file)



# process both folders
directory_query1 = "image_without_background"
directory = 'image_without_background'
directory_bbdd = 'data/BBDD/week2'
print("Current working directory:", os.getcwd())
print("Processing directory 1:")
#process_directory(directory_query1)

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

# Define the histogram keys and distance functions
histogram_keys = [
    'hist_LAB_32_2D','hist_HSV_8_1D','hist_HSV_8_3D',
    'hist_resize_LAB_64_1D','hist_resize_HSV_64_1D']

distance_functions = {
    'our_metric': utils.our_metric,
    'X2_distance': utils.X2_distance,
    'hellinger_kernel': utils.hellinger_kernel
}
results_df = pd.DataFrame(columns=['hist_key', 'metric_name', 'k1', 'k5'])
# Loop over each histogram key
for hist_key in histogram_keys:
    print(f"Processing histogram key: {hist_key}")
    
    # Loop over each distance function
    for metric_name, metric_func in distance_functions.items():
        print(f"  Using distance function: {metric_name}")

        list_results_k_1 = []
        list_results_k_5 = []

        files = os.listdir(directory)
        files_sorted = sorted(files)

        # Loop over each file to compare
        for file_compare_image in files_sorted:
            if file_compare_image.endswith('_w2.pkl') and file_compare_image != 'gt_corresps.pkl':
                pkl_grey_path = os.path.join(directory, file_compare_image)
                with open(pkl_grey_path, 'rb') as pkl_file:
                    histograms_first = pickle.load(pkl_file)

                distances = []
                index_qsd1_w1 = extract_number_from_filename_qsd1_w1(file_compare_image)

                # Loop over each file in the database
                for filename in os.listdir(directory_bbdd):
                    if filename.endswith('_w2.pkl') and filename != 'relationships.pkl':
                        pkl_path = os.path.join(directory_bbdd, filename)

                        with open(pkl_path, 'rb') as pkl_file:
                            histograms = pickle.load(pkl_file)

                        # Calculate distance for the current histogram key
                        histogram_first = histograms_first[hist_key]
                        histogram = histograms[hist_key]

                        # Calculate the distance
                        distance = metric_func(histogram_first, histogram)
                        index = extract_number_from_filename(filename)

                        distances.append((distance, index))

                # Sort the distances and select the top k results
                distances.sort(key=lambda x: x[0])  # Sort by distance (lowest first)
                top_k_1_result = [distances[0][1]]
                top_k_5_results = [index for _, index in distances[:5]]

                # Save the indices in a list
                list_results_k_1.append(top_k_1_result)
                list_results_k_5.append(top_k_5_results)

        # Flatten the results for comparison with ground truth
        predicted_flattened_k_5 = [p for p in list_results_k_5]

        with open(f'datasets/qsd2_w1/gt_corresps.pkl', 'rb') as f:
            ground_truth = pickle.load(f)

        # Print the MAP@k results for the current histogram key with the current distance function
        mapk_k1 = utils.mapk(ground_truth, list_results_k_1, k=1)
        mapk_k5 = utils.mapk(ground_truth, predicted_flattened_k_5, k=5)

        temp_df = pd.DataFrame({
            'hist_key': [hist_key],
            'metric_name': [metric_name],
            'k1': [mapk_k1],
            'k5': [mapk_k5]
        })

        # Concatenar los resultados al DataFrame principal
        results_df = pd.concat([results_df, temp_df], ignore_index=True)

        print(f"    {hist_key} with {metric_name}; k=1 mapk: {mapk_k1}")
        print(f"    {hist_key} with {metric_name}; k=5 mapk: {mapk_k5}")
# Ordenar el DataFrame por el valor de k=1 en orden descendente
results_df = results_df.sort_values(by='k1', ascending=False)

# Guardar el DataFrame en un archivo CSV
results_df.to_csv('mapk_results_withoutback.csv', index=False)
print("Finished processing all histogram keys and distance functions.")