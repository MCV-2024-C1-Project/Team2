import os
import pickle
import re
import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import utils

# Function to extract the image number from the filename
def extract_number_from_filename(filename):
    match = re.search(r'bbdd_(\d{5})_w4\.pkl', filename)
    if match:
        return int(match.group(1))

# Function to find bidirectional matches between two sets of descriptors using the ratio test
def find_bidirectional_matches(desc1, desc2, ratio_thresh=0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # Use crossCheck=False to manually control bidirectional matching
    
    # Matches from desc1 to desc2
    matches1 = bf.knnMatch(desc1, desc2, k=2)
    # Matches from desc2 to desc1
    matches2 = bf.knnMatch(desc2, desc1, k=2)
    good_matches = []

    # Apply the ratio test and check for bidirectional matching
    for i, match_pair in enumerate(matches1):
        if len(match_pair) == 2:  # Ensure there are at least two matches
            m, n = match_pair
            # Apply Lowe's ratio test
            if m.distance < ratio_thresh * n.distance:
                # Verify if the reverse match is also valid
                for match_pair in matches2:
                    if len(match_pair) == 2:  # Ensure the tuple has two elements
                        m2, n2 = match_pair  # Unpack the tuple
                        if m2.queryIdx == m.trainIdx and m2.trainIdx == m.queryIdx:
                            good_matches.append(m)
                            break

    return good_matches

# Directories for the query and database folders
directory_query = 'filtered_cropped_qst1_w4'
directory_bbdd = '../../data/BBDD/week4'

files = os.listdir(directory_query)  # List of files in the query directory
files_sorted = sorted(files)
list_results_k_10 = []  # List to store the top k=10 results

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
        contour_results_k_10 = []

        # Process each contour file
        for contour_path in contour_paths:
            if not os.path.exists(contour_path):
                continue

            with open(contour_path, 'rb') as pkl_file:
                query_desc = pickle.load(pkl_file)
            # print(contour_path)

            good_matches_list = []  # List of matches for each image in the database
            best_match_index = None

            # Loop over each file in the database
            for file_db in os.listdir(directory_bbdd):
                if file_db.endswith('_w4.pkl') and file_db != 'relationships.pkl':
                    db_path = os.path.join(directory_bbdd, file_db)
                    with open(db_path, 'rb') as pkl_file:
                        db_desc = pickle.load(pkl_file)

                    # Compare descriptors
                    matches = find_bidirectional_matches(query_desc['desc_orb'], db_desc['desc_orb'], 0.8)

                    good_matches_list.append((len(matches), extract_number_from_filename(file_db)))

            # Sort images by the number of good matches in descending order
            good_matches_list.sort(reverse=True, key=lambda x: x[0])
            good_10_matches = [match[1] for match in good_matches_list[:10]]

            # Check if the difference between the top matches is significant
            if len(good_matches_list) > 1:
                first_match = good_matches_list[0]
                second_match = good_matches_list[1]
                third_match = good_matches_list[2] if len(good_matches_list) > 2 else None
                fourth_match = good_matches_list[3] if len(good_matches_list) > 3 else None

                if second_match and (first_match[0] - second_match[0]) > (first_match[0] / 3):
                    best_match_index = good_10_matches
                elif third_match and (first_match[0] - third_match[0]) > (first_match[0] / 3):
                    best_match_index = good_10_matches
                elif fourth_match and (first_match[0] - fourth_match[0]) > (first_match[0] / 3):
                    best_match_index = good_10_matches
                else:
                    # If differences are not significant, predict as -1
                    best_match_index = [-1]
            else:
                best_match_index = [-1]  # If not enough images compared, predict as -1

            top_k_10_results = best_match_index[:10]
            # Append the results for this contour
            contour_results_k_10.append(top_k_10_results)

        # Add contour results to the main list
        if len(contour_results_k_10) == 1:
            list_results_k_10.append([contour_results_k_10[0]])
        else:
            list_results_k_10.append(contour_results_k_10)

print(list_results_k_10)

# Save the results to a CSV file
with open('../../results/week4/QST1/method1/result.pkl', 'wb') as f:
    pickle.dump(list_results_k_10, f)
print("Results.pkl saved successfully")
