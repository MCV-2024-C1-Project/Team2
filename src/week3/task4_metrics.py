import re
import os
import pickle
import pandas as pd
import utils

directory = 'filtered_cropped_qsd2_w3'  # Directory containing the filtered images
directory_bbdd = '../../data/BBDD/week3'  # Directory containing the database of images

def extract_number_from_filename(filename):
    '''Function to extract the number of the image from the filename'''
    match = re.search(r'bbdd_(\d{5})_w3\.pkl', filename)
    if match:
        return int(match.group(1))

def extract_number_from_filename_qsd1_w1(filename):
    '''Function to extract the number of the image from the filename'''
    match = re.search(r'(\d{5})_w3\.pkl', filename)
    if match:
        return int(match.group(1))

# Define the histogram keys and distance functions
histogram_keys = [
                'hist_DCT_ycbcr_n32_c10_1D',
                'hist_LBP_LAB_n8_r2_1D','hist_LBPM_LAB_n8_1D','hist_DCT_HSV_n32_c10_1D',
]

# Dictionary of distance functions
distance_functions = {
    #'our_metric':utils.our_metric,  # Custom distance metric
    'X2_distance': utils.X2_distance,  # Chi-squared distance
    'L1_dist': utils.L1_dist,  # L1 distance (Manhattan)
    #'euc_dist': utils.euc_dist,  # Euclidean distance
    #'histogram_similarity':utils.histogram_similarity,  # Histogram similarity metric (commented out)
    #'hellinger_kernel': utils.hellinger_kernel  # Hellinger distance
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
                        histograms_first = pickle.load(pkl_file)  # Load histograms for this contour

                    distances = []  # List to store distances
                    index_qsd1_w1 = extract_number_from_filename_qsd1_w1(file_compare_image)  # Extract index from filename

                    # Loop over each file in the database
                    for filename in os.listdir(directory_bbdd):
                        if filename.endswith('_w3.pkl') and filename != 'relationships.pkl':
                            pkl_path = os.path.join(directory_bbdd, filename)  # Full path to the database file

                            with open(pkl_path, 'rb') as pkl_file:
                                histograms = pickle.load(pkl_file)  # Load histograms from the database file

                            # Calculate distance for the current histogram key
                            histogram_first = histograms_first[hist_key]
                            histogram = histograms[hist_key]

                            # Calculate the distance using the selected metric
                            distance = metric_func(histogram_first, histogram)
                            index = extract_number_from_filename(filename)  # Extract index from database filename

                            distances.append((distance, index))  # Append distance and index to the list

                    # Sort the distances and select the top k results
                    distances.sort(key=lambda x: x[0])  # Sort by distance (lowest first)
                    top_k_1_result = [distances[0][1]]  # Top 1 result
                    top_k_5_results = [index for _, index in distances[:5]]  # Top 5 results

                    # Append the results for this contour
                    contour_results_k_1.append(top_k_1_result)
                    contour_results_k_5.append(top_k_5_results)
                    #print(f'result.... '+contour_path)
                    #print(contour_results_k_1)

                # Add contour results to the main lists (conditional for one or two contours)
                if len(contour_results_k_1) == 1:
                    list_results_k_1.append(contour_results_k_1[0])  # Single painting, add directly
                    list_results_k_5.append(contour_results_k_5[0])
                else:
                    list_results_k_1.append([result[0] for result in contour_results_k_1])  # Multiple paintings, add as list of lists
                    list_results_k_5.append(contour_results_k_5)

        # Flatten the results for comparison with ground truth
        predicted_flattened_k_5 = [p for p in list_results_k_5]

        with open('../../datasets/qsd2_w3/gt_corresps.pkl', 'rb') as f:
            ground_truth = pickle.load(f)  # Load ground truth data

        print(list_results_k_1)
        print(predicted_flattened_k_5)
        # Print the MAP@k results for the current histogram key with the current distance function
        mapk_k1 = utils.mapk(ground_truth, list_results_k_1, k=1)  # Compute mean average precision at k=1
        mapk_k5 = utils.mapk(ground_truth, predicted_flattened_k_5, k=5)  # Compute mean average precision at k=5

        temp_df = pd.DataFrame({
            'hist_key': [hist_key],
            'metric_name': [metric_name],
            'k1': [mapk_k1],
            'k5': [mapk_k5]
        })

        # Concatenate the results into the main DataFrame
        results_df = pd.concat([results_df, temp_df], ignore_index=True)

        print(f"    {hist_key} with {metric_name}; k=1 mapk: {mapk_k1}")
        print(f"    {hist_key} with {metric_name}; k=5 mapk: {mapk_k5}")


# Sort the DataFrame by the value of k=1 in descending order
results_df = results_df.sort_values(by='k1', ascending=False)

# Save the DataFrame to a CSV file
results_df.to_csv('mapk_results_week3_qsd2.csv', index=False)
print("Finished processing all histogram keys and distance functions.")



# for hist_key in histogram_keys:
#     print(f"Processing histogram key: {hist_key}")
    
#     # Loop over each distance function
#     for metric_name, metric_func in distance_functions.items():
#         print(f"  Using distance function: {metric_name}")

#         list_results_k_1 = []  # List to store top k=1 results
#         list_results_k_5 = []  # List to store top k=5 results

#         files = os.listdir(directory)  # List of files in the directory
#         files_sorted = sorted(files)  # Sort the files

#         # Loop over each file to compare
#         for file_compare_image in files_sorted:
#             if file_compare_image.endswith('_w3.pkl') and file_compare_image != 'gt_corresps.pkl':
#                 # Check if this is a multi-painting file and load accordingly
#                 if file_compare_image.endswith('_contour1_w3.pkl'):
#                     # Caso donde la imagen tiene una o dos pinturas
#                     contour_paths = [os.path.join(directory, file_compare_image)]
                    
#                     # Verificar si también existe el archivo de la segunda pintura
#                     contour2_path = file_compare_image.replace('_contour1_w3.pkl', '_contour2_w3.pkl')
#                     if os.path.exists(os.path.join(directory, contour2_path)):
#                         contour_paths.append(os.path.join(directory, contour2_path))
#                 else:
#                     # En caso de que el archivo no tenga el sufijo esperado
#                     contour_paths = [os.path.join(directory, file_compare_image)]

#                 # Prepare to collect results for multiple contours if they exist
#                 contour_results_k_1 = []
#                 contour_results_k_5 = []

#                 # Process each contour file
#                 for contour_path in contour_paths:
#                     if not os.path.exists(contour_path):
#                         continue

#                     with open(contour_path, 'rb') as pkl_file:
#                         histograms_first = pickle.load(pkl_file)  # Load histograms for this contour

#                     distances = []  # List to store distances
#                     index_qsd1_w1 = extract_number_from_filename_qsd1_w1(file_compare_image)  # Extract index from filename

#                     # Loop over each file in the database
#                     for filename in os.listdir(directory_bbdd):
#                         if filename.endswith('_w3.pkl') and filename != 'relationships.pkl':
#                             pkl_path = os.path.join(directory_bbdd, filename)  # Full path to the database file

#                             with open(pkl_path, 'rb') as pkl_file:
#                                 histograms = pickle.load(pkl_file)  # Load histograms from the database file

#                             # Calculate distance for the current histogram key
#                             histogram_first = histograms_first[hist_key]
#                             histogram = histograms[hist_key]

#                             # Calculate the distance using the selected metric
#                             distance = metric_func(histogram_first, histogram)
#                             index = extract_number_from_filename(filename)  # Extract index from database filename

#                             distances.append((distance, index))  # Append distance and index to the list

#                     # Sort the distances and select the top k results
#                     distances.sort(key=lambda x: x[0])  # Sort by distance (lowest first)
#                     top_k_1_result = [distances[0][1]]  # Top 1 result
#                     top_k_5_results = [index for _, index in distances[:5]]  # Top 5 results

#                     # Append the results for this contour
#                     contour_results_k_1.append(top_k_1_result)
#                     contour_results_k_5.append(top_k_5_results)

#                 # Add contour results to the main lists (each contour is a separate prediction)
#                 list_results_k_1.append(contour_results_k_1)
#                 list_results_k_5.append(contour_results_k_5)

#         # Flatten the results for comparison with ground truth
#         predicted_flattened_k_5 = [p for p in list_results_k_5]

#         with open('../../datasets/qsd2_w3/gt_corresps.pkl', 'rb') as f:
#             ground_truth = pickle.load(f)  # Load ground truth data

#         print(list_results_k_1)
#         print(list_results_k_5)
#         # Print the MAP@k results for the current histogram key with the current distance function
#         mapk_k1 = utils.mapk(ground_truth, list_results_k_1, k=1)  # Compute mean average precision at k=1
#         mapk_k5 = utils.mapk(ground_truth, predicted_flattened_k_5, k=5)  # Compute mean average precision at k=5

#         temp_df = pd.DataFrame({
#             'hist_key': [hist_key],
#             'metric_name': [metric_name],
#             'k1': [mapk_k1],
#             'k5': [mapk_k5]
#         })

#         # Concatenate the results into the main DataFrame
#         results_df = pd.concat([results_df, temp_df], ignore_index=True)

#         print(f"    {hist_key} with {metric_name}; k=1 mapk: {mapk_k1}")
#         print(f"    {hist_key} with {metric_name}; k=5 mapk: {mapk_k5}")
