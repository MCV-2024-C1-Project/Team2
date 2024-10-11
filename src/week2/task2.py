import re
import os
import pickle

import utils

directory = 'datasets/qsd1_w1/week2'
directory_bbdd = 'data/BBDD/week2'

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
    'hist_grey_8', 'hist_grey_128', 'hist_grey_256',
    'hist_RGB_8', 'hist_RGB_128', 'hist_RGB_256',
    'hist_LAB_8', 'hist_LAB_128', 'hist_LAB_256',
    'hist_HSV_8', 'hist_HSV_128', 'hist_HSV_256'
]

distance_functions = {
    'our_metric': utils.our_metric,
    'X2_distance': utils.X2_distance,
    'L1_dist': utils.L1_dist,
    'euc_dist': utils.euc_dist,
    'histogram_similarity': utils.histogram_similiarity,
    'hellinger_kernel': utils.hellinger_kernel
}

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

        with open(f'datasets/qsd1_w1/gt_corresps.pkl', 'rb') as f:
            ground_truth = pickle.load(f)

        # Print the MAP@k results for the current histogram key with the current distance function
        mapk_k1 = utils.mapk(ground_truth, list_results_k_1, k=1)
        mapk_k5 = utils.mapk(ground_truth, predicted_flattened_k_5, k=5)
        print(f"    {hist_key} with {metric_name}; k=1 mapk: {mapk_k1}")
        print(f"    {hist_key} with {metric_name}; k=5 mapk: {mapk_k5}")

print("Finished processing all histogram keys and distance functions.")