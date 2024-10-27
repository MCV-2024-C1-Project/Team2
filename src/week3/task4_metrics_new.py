import re
import os
import pickle
import pandas as pd
import utils
import numpy as np

directory = 'filtered_cropped_qsd2_w3'  # Directory containing the filtered images
directory_bbdd = '../../data/BBDD/week3'  # Directory containing the database of images


def extract_number_from_filename(filename):
    '''Function to extract the number of the image from the filename'''
    match = re.search(r'bbdd_(\d{5})_w3\.pkl', filename)
    if match:
        return int(match.group(1))


def extract_number_from_filename_qsd2(filename):
    '''Function to extract the number of the image from the filename'''
    match = re.search(r'(\d{5})_w3\.pkl', filename)
    if match:
        return int(match.group(1))


def custom_mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k for cases with one or two paintings.
    actual: List of lists containing the ground truth for each query.
    predicted: List of lists containing the predicted results for each query.
    k: The maximum number of predicted elements to consider.
    """
    def apk_single(actual, predicted, k):
        """
        Computes the average precision at k for a single painting.
        actual: The ground truth list with a single element.
        predicted: The predicted list.
        k: The maximum number of predicted elements to consider.
        """
        if len(predicted) > k:
            predicted = predicted[:k]

        return 1.0 if actual[0] in predicted else 0.0

    def apk_double(actual, predicted, k):
        """
        Computes the average precision at k for two paintings.
        actual: The ground truth list with two elements.
        predicted: The list of predicted lists for each painting.
        k: The maximum number of predicted elements to consider.
        """
        # Ensure that predicted is a list of two lists
        if not isinstance(predicted[0], list):
            return 0.0

        if len(predicted[0]) > k:
            predicted = [pred[:k] for pred in predicted]

        # Check if each ground truth value appears in its corresponding predicted list
        return 1.0 if actual[0] in predicted[0] and actual[1] in predicted[1] else 0.0

    # Calculate mean of average precision at k for all queries
    scores = []
    for a, p in zip(actual, predicted):
        if len(a) == 1:  # Single painting case
            scores.append(apk_single(a, p, k))
        elif len(a) == 2:  # Two paintings case
            scores.append(apk_double(a, p, k))

    return np.mean(scores)


# Define the histogram keys and distance functions
histogram_keys = [
    'hist_DCT_ycbcr_n32_c10_1D',
    'hist_LBP_LAB_n8_r2_1D', 
    'hist_LBPM_LAB_n8_1D', 
    'hist_DCT_HSV_n32_c10_1D'
]

# Dictionary of distance functions
distance_functions = {
    'X2_distance': utils.X2_distance,
    'L1_dist': utils.L1_dist
}

# DataFrame to store results
results_df = pd.DataFrame(columns=['hist_key', 'metric_name', 'k1', 'k5'])

# Iterate over histogram keys
for hist_key in histogram_keys:
    print(f"Processing histogram key: {hist_key}")

    # Loop over each distance function
    for metric_name, metric_func in distance_functions.items():
        print(f"  Using distance function: {metric_name}")

        list_results_k_1 = []  # List to store top k=1 results
        list_results_k_5 = []  # List to store top k=5 results

        files = os.listdir(directory)  # List of files in the directory
        files_sorted = sorted(files)  # Sort the files

        # Loop over each file to compare
        for file_compare_image in files_sorted:
            if file_compare_image.endswith('_w3.pkl') and file_compare_image != 'gt_corresps.pkl':
                # Check if this is a multi-painting file and load accordingly
                contour_paths = [os.path.join(directory, file_compare_image)]
                if file_compare_image.endswith('_contour1_w3.pkl'):
                    contour2_path = file_compare_image.replace('_contour1_w3.pkl', '_contour2_w3.pkl')
                    if os.path.exists(os.path.join(directory, contour2_path)):
                        contour_paths.append(os.path.join(directory, contour2_path))
                if file_compare_image.endswith('_contour2_w3.pkl'):
                    continue

                # Prepare to collect results for multiple contours if they exist
                contour_results_k_1 = []
                contour_results_k_5 = []

                # Process each contour file
                for contour_path in contour_paths:
                    if not os.path.exists(contour_path):
                        continue

                    with open(contour_path, 'rb') as pkl_file:
                        histograms_first = pickle.load(pkl_file)

                    distances = []  # List to store distances
                    index_qsd2 = extract_number_from_filename_qsd2(file_compare_image)

                    # Loop over each file in the database
                    for filename in os.listdir(directory_bbdd):
                        if filename.endswith('_w3.pkl') and filename != 'relationships.pkl':
                            pkl_path = os.path.join(directory_bbdd, filename)

                            with open(pkl_path, 'rb') as pkl_file:
                                histograms = pickle.load(pkl_file)

                            # Calculate distance for the current histogram key
                            histogram_first = histograms_first[hist_key]
                            histogram = histograms[hist_key]

                            # Calculate the distance using the selected metric
                            distance = metric_func(histogram_first, histogram)
                            index = extract_number_from_filename(filename)

                            distances.append((distance, index))

                    # Sort the distances and select the top k results
                    distances.sort(key=lambda x: x[0])
                    top_k_1_result = [distances[0][1]]
                    top_k_5_results = [index for _, index in distances[:5]]

                    # Append the results for this contour
                    contour_results_k_1.append(top_k_1_result)
                    contour_results_k_5.append(top_k_5_results)

                # Add contour results to the main lists
                if len(contour_results_k_1) == 1:
                    list_results_k_1.append(contour_results_k_1[0])
                    list_results_k_5.append(contour_results_k_5[0])
                else:
                    list_results_k_1.append([result[0] for result in contour_results_k_1])
                    list_results_k_5.append(contour_results_k_5)

        # Flatten the results for comparison with ground truth
        flattened_k1 = [item for sublist in list_results_k_1 for item in sublist]
        flattened_k5 = [item for sublist in list_results_k_5 for item in sublist]

        # Load the ground truth
        with open('../../datasets/qsd2_w3/gt_corresps.pkl', 'rb') as f:
            ground_truth = pickle.load(f)

        print("ground_truth:\n", ground_truth)
        print("list_results_k_1:\n", list_results_k_1)
        print("list_results_k_5:\n", list_results_k_5)

        # Compute the MAP@k results
        mapk_k1 = custom_mapk(ground_truth, list_results_k_1, k=1)
        mapk_k5 = custom_mapk(ground_truth, list_results_k_5, k=5)

        # Save the results to the DataFrame
        temp_df = pd.DataFrame({
            'hist_key': [hist_key],
            'metric_name': [metric_name],
            'k1': [mapk_k1],
            'k5': [mapk_k5]
        })
        results_df = pd.concat([results_df, temp_df], ignore_index=True)

        print(f"    {hist_key} with {metric_name}; k=1 mapk: {mapk_k1}")
        print(f"    {hist_key} with {metric_name}; k=5 mapk: {mapk_k5}")

# Save the results to a CSV file
results_df = results_df.sort_values(by='k1', ascending=False)
results_df.to_csv('mapk_results_corrected.csv', index=False)
print("Finished processing all histogram keys and distance functions.")
