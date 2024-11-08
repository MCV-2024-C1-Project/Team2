import os
import pickle
import re
import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import utils

# Function to calculate mAP@k
def custom_mapk(actual, predicted, k=10):
    # Helper function to calculate AP@k for a single prediction
    def apk_single(actual, predicted, k):
        if len(predicted) > k:
            predicted = predicted[:k]
        return 1.0 if actual[0] in predicted else 0.0

    # Helper function to calculate AP@k for double predictions (e.g., two correct answers)
    def apk_double(actual, predicted, k):
        if isinstance(predicted[0], list):
            # Ensure k-limit for nested lists
            if len(predicted[0]) > k:
                predicted = [pred[:k] for pred in predicted]
            
            # Safe checks to avoid IndexError
            actual_first = actual[0] if len(actual) > 0 else None
            actual_second = actual[1] if len(actual) > 1 else None
            predicted_first = predicted[0][0] if len(predicted[0]) > 0 else None
            predicted_second = predicted[1][0] if len(predicted) > 1 and len(predicted[1]) > 0 else None

            if actual_first == predicted_first and actual_second == predicted_second:
                return 1.0
            elif actual_first == predicted_first or actual_second == predicted_second:
                return 0.5
            else:
                return 0.0
        else:
            # Safe checks to avoid IndexError
            actual_first = actual[0] if len(actual) > 0 else None
            actual_second = actual[1] if len(actual) > 1 else None
            predicted_first = predicted[0] if len(predicted) > 0 else None
            predicted_second = predicted[1] if len(predicted) > 1 else None

            if actual_first == predicted_first and actual_second == predicted_second:
                return 1.0
            elif actual_first == predicted_first or actual_second == predicted_second:
                return 0.5
            else:
                return 0.0

    scores = []
    for a, p in zip(actual, predicted):
        if len(a) == 1:
            scores.append(apk_single(a, p, k))
        elif len(a) == 2:
            scores.append(apk_double(a, p, k))
    return np.mean(scores)

# Function to calculate mAP@k for multiple k values
def calculate_mapk(matches, ground_truth, k_values=1):
    results = {}
    results = float(custom_mapk(ground_truth, matches, k_values))
    return results

# Function to extract the image number from the filename
def extract_number_from_filename(filename):
    match = re.search(r'bbdd_(\d{5})_w4\.pkl', filename)
    if match:
        return int(match.group(1))

# Function to find bidirectional matches between two sets of descriptors
def find_bidirectional_matches(desc1, desc2, ratio_thresh=0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # Use crossCheck=False for manual bidirectional matching

    # Matches from desc1 to desc2
    matches1 = bf.knnMatch(desc1, desc2, k=2)
    # Matches from desc2 to desc1
    matches2 = bf.knnMatch(desc2, desc1, k=2)
    good_matches = []

    # Apply ratio test and check bidirectional match
    for i, match_pair in enumerate(matches1):
        if len(match_pair) == 2:  # Ensure there are at least two matches
            m, n = match_pair
            # Apply Lowe's ratio test
            if m.distance < ratio_thresh * n.distance:
                # Check if the reverse match is also valid
                for match_pair in matches2:
                    if len(match_pair) == 2:  # Ensure the tuple has two elements
                        m2, n2 = match_pair  # Unpack the tuple
                        if m2.queryIdx == m.trainIdx and m2.trainIdx == m.queryIdx:
                            good_matches.append(m)
                            break

    return good_matches

# Directories for query and database folders
directory_query = 'filtered_cropped_qsd1_w4'
directory_bbdd = '../../data/BBDD/week4'

# Provided ground truth
file_path = '../../datasets/qsd1_w4/gt_corresps.pkl'
# Load the .pkl file
with open(file_path, 'rb') as file:
    data = pickle.load(file)
ground_truth = data

# Define ratio thresholds range
ratio_thresholds = np.arange(0.8, 0.95, 0.05)  # Ratio values from 0.8 to 0.95

# Array to store metrics and predictions
all_metrics = []
results_df = pd.DataFrame(columns=['descriptor', 'metric_name', 'ratio_threshold', 'k1', 'k5'])

# Iterate over different ratio_thresh values
for ratio_thresh in ratio_thresholds:
    files = os.listdir(directory_query)  # List of files in the directory
    files_sorted = sorted(files) 
    list_results_k_1 = []  # List to store top k=1 results
    list_results_k_5 = [] 
    for file_compare_image in files_sorted:
                if file_compare_image.endswith('_w4.pkl') and file_compare_image != 'gt_corresps.pkl':
                    # Check if this is a multi-painting file and load accordingly
                    contour_paths = [os.path.join(directory_query, file_compare_image)]
                    if file_compare_image.endswith('_contour1_w4.pkl'):
                        contour2_path = file_compare_image.replace('_contour1_w4.pkl', '_contour2_w4.pkl')
                        if os.path.exists(os.path.join(directory_query, contour2_path)):
                            contour_paths.append(os.path.join(directory_query, contour2_path))
                    if file_compare_image.endswith('_contour2_w4.pkl'):
                        continue

                    # Prepare to collect results for multiple contours if they exist
                    contour_results_k_1 = []
                    contour_results_k_5 = []

                    # Process each contour file
                    for contour_path in contour_paths:
                        if not os.path.exists(contour_path):
                            continue

                        with open(contour_path, 'rb') as pkl_file:
                            query_desc = pickle.load(pkl_file)
                        #print(contour_path)

                        good_matches_list = []  # List of matches for each image in the database
                        best_match_index = None
                        # Loop over each file in the database
                        for file_db in os.listdir(directory_bbdd):
                            if file_db.endswith('_w4.pkl') and file_db != 'relationships.pkl':
                                db_path = os.path.join(directory_bbdd, file_db)
                                with open(db_path, 'rb') as pkl_file:
                                    db_desc = pickle.load(pkl_file)

                                # Compare the ORB descriptors
                                matches = find_bidirectional_matches(query_desc['desc_orb'], db_desc['desc_orb'], ratio_thresh)

                                good_matches_list.append((len(matches), extract_number_from_filename(file_db)))

                        # Sort the images by the number of good_matches in descending order
                        good_matches_list.sort(reverse=True, key=lambda x: x[0])
                        good_5_matches = [match[1] for match in good_matches_list[:5]]
                        # Check if the difference is significant between the top matches
                        if len(good_matches_list) > 1:
                            first_match = good_matches_list[0]
                            second_match = good_matches_list[1]
                            third_match = good_matches_list[2] if len(good_matches_list) > 2 else None
                            fourth_match = good_matches_list[3] if len(good_matches_list) > 3 else None

                            if second_match and (first_match[0] - second_match[0]) > (first_match[0] / 3):
                                best_match_index = good_5_matches
                            elif third_match and (first_match[0] - third_match[0]) > (first_match[0] / 3):
                                best_match_index = good_5_matches
                            elif fourth_match and (first_match[0] - fourth_match[0]) > (first_match[0] / 3):
                                best_match_index = good_5_matches
                            else:
                                # If the differences are not significant, predict as -1
                                best_match_index = [-1]
                        else:
                            best_match_index = [-1]  # If there aren't enough images compared, predict as -1


                        top_k_1_result = [best_match_index[0]]  # Take the first element of the list as k=1
                        top_k_5_results = best_match_index[:5] 
                        # Append the results for this contour
                        contour_results_k_1.append(top_k_1_result)
                        contour_results_k_5.append(top_k_5_results)

                    # Add contour results to the main lists
                    if len(contour_results_k_1) == 1:
                        list_results_k_1.append(contour_results_k_1[0])
                        list_results_k_5.append([contour_results_k_5[0]])
                    else:
                        list_results_k_1.append([result[0] for result in contour_results_k_1])
                        list_results_k_5.append(contour_results_k_5)
                                    

    print(list_results_k_1)
    print(list_results_k_5)

    print("ground_truth:\n", ground_truth)
    print("list_results_k_1:\n", list_results_k_1)
    print("list_results_k_5:\n", list_results_k_5)

    # Compute the MAP@k results
    mapk_k1 = calculate_mapk(list_results_k_1,ground_truth, k_values=1)
    mapk_k5 = calculate_mapk(list_results_k_5,ground_truth, k_values=5)

    # Save the results to the DataFrame
    temp_df = pd.DataFrame({
        'descriptor': ['ORB'],
        'metric_name': ['L2'],
        'ratio_treshold': [ratio_thresh],
        'k1': [mapk_k1],
        'k5': [mapk_k5]})
    results_df = pd.concat([results_df, temp_df], ignore_index=True)

    print(f"    ORB with L2; ratio_treshold {ratio_thresh}, k=1 mapk: {mapk_k1}")
    print(f"    ORB with L2; ratio_treshold {ratio_thresh}, k=5 mapk: {mapk_k5}")
# Save the results to a CSV file
results_df = results_df.sort_values(by='k1', ascending=False)
results_df.to_csv('mapk_results.csv', index=False)
print("Finished processing all histogram keys and distance functions.")
