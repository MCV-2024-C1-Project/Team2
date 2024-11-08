import os
import pickle
import re
import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to extract the image number from the filename
def extract_number_from_filename(filename):
    match = re.search(r'bbdd_(\d{5})_w4\.pkl', filename)
    if match:
        return int(match.group(1))

# Function to find matches using BFMatcher and ratio test
def find_matches_SIFT(desc1, desc2, ratio_thresh=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
        
    # Sort matches by distance (ascending order)
    matches = sorted(matches, key=lambda x: x.distance)
        
    good_matches = []
    
    # Collect matches that pass the ratio test
    good_matches = [m for m in matches if m.distance < ratio_thresh]
    return good_matches

# Function to find bidirectional matches for SIFT descriptors
def find_bidirectional_matches_SIFT(desc1, desc2, ratio_thresh=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # Use crossCheck=False for manual bidirectional matching
    
    # Matches from desc1 to desc2
    matches1 = bf.knnMatch(desc1, desc2, k=2)
    # Matches from desc2 to desc1
    matches2 = bf.knnMatch(desc2, desc1, k=2)
    good_matches = []

    # Apply ratio test and verify bidirectional matching
    for i, match_pair in enumerate(matches1):
        if len(match_pair) == 2:  # Ensure we have at least two matches
            m, n = match_pair
            # Apply Lowe's ratio test
            if m.distance < ratio_thresh * n.distance:
                # Check if the reverse match is also valid
                for m2, n2 in matches2:
                    if m2.queryIdx == m.trainIdx and m2.trainIdx == m.queryIdx:
                        good_matches.append(m)
                        break  # Only add the match once

    return good_matches

# Function to find bidirectional matches for ORB descriptors
def find_bidirectional_matches_ORB(desc1, desc2, ratio_thresh=0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # Use crossCheck=False for manual bidirectional matching
    
    # Matches from desc1 to desc2
    matches1 = bf.knnMatch(desc1, desc2, k=2)
    # Matches from desc2 to desc1
    matches2 = bf.knnMatch(desc2, desc1, k=2)
    good_matches = []

    # Apply ratio test and verify bidirectional matching
    for i, match_pair in enumerate(matches1):
        if len(match_pair) == 2:  # Ensure we have at least two matches
            m, n = match_pair
            # Apply Lowe's ratio test
            if m.distance < ratio_thresh * n.distance:
                # Check if the reverse match is also valid
                for m2, n2 in matches2:
                    if m2.queryIdx == m.trainIdx and m2.trainIdx == m.queryIdx:
                        good_matches.append(m)
                        break  # Only add the match once

    return good_matches

# Function to run SIFT descriptor matching
def descriptor_SIFT(directory_query, directory_bbdd, files_query, GT):
    ratio_thresholds = np.arange(0.80, 0.90, 0.05)  # Range of ratio thresholds from 0.8 to 0.95

    # Array to store metrics and predictions
    all_metrics = []

    # Iterate over different ratio_thresh values
    for ratio_thresh in ratio_thresholds:
        predicted_matches = []  # Reset predictions for each ratio_thresh

        # Iterate over each query image
        for idx, file_query in enumerate(files_query):
            # Load the query image and its SIFT descriptor
            query_path = os.path.join(directory_query, file_query)
            with open(query_path, 'rb') as pkl_file:
                query_desc = pickle.load(pkl_file)

            good_matches_list = []  # List of matches for each database image
            best_match_index = None

            # Compare the query with all database images
            for file_db in os.listdir(directory_bbdd):
                if file_db.endswith('_w4.pkl') and file_db != 'relationships.pkl':
                    db_path = os.path.join(directory_bbdd, file_db)
                    with open(db_path, 'rb') as pkl_file:
                        db_desc = pickle.load(pkl_file)

                    matches = find_bidirectional_matches_SIFT(query_desc['desc_sift'], db_desc['desc_sift'], ratio_thresh)

                    good_matches_list.append((len(matches), extract_number_from_filename(file_db)))

            # Sort images by number of good_matches in descending order
            good_matches_list.sort(reverse=True, key=lambda x: x[0])

            # Check if the difference between top matches is significant
            if len(good_matches_list) > 1:
                first_match = good_matches_list[0]
                second_match = good_matches_list[1]
                third_match = good_matches_list[2] if len(good_matches_list) > 2 else None
                fourth_match = good_matches_list[3] if len(good_matches_list) > 3 else None

                if second_match and (first_match[0] - second_match[0]) > (first_match[0] / 3):
                    best_match_index = first_match[1]
                elif third_match and (first_match[0] - third_match[0]) > (first_match[0] / 3):
                    best_match_index = first_match[1]
                elif fourth_match and (first_match[0] - fourth_match[0]) > (first_match[0] / 3):
                    best_match_index = first_match[1]
                else:
                    # Predict -1 if differences are not significant
                    best_match_index = -1
            else:
                best_match_index = -1  # Predict -1 if not enough images are compared

            # Add prediction for the current image
            predicted_matches.append(best_match_index)

        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        TP = sum(1 for p, g in zip(predicted_matches, GT) if p == g)
        FP = sum(1 for p, g in zip(predicted_matches, GT) if p != g and p not in GT)
        FN = sum(1 for g, p in zip(GT, predicted_matches) if g != p and g not in predicted_matches)

        # Calculate Precision, Recall, and F1 Score manually
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        # Store metrics for this threshold combination
        all_metrics.append((precision, recall, f1, ratio_thresh))

        # Print progress
        print(f"Processed query {idx + 1}/{len(files_query)} | Predicted: {best_match_index} | F1: {f1} | Recall: {recall} | Precision: {precision}")

    # Final evaluation (no iteration over ratio_thresh or min_match_thresh)
    best_metric = max(all_metrics, key=lambda x: x[2])  # Find the tuple with the highest F1
    best_precision, best_recall, best_f1, best_ratio_thresh = best_metric

    # Print the result with the best ratio_thresh
    print(f"\nBest F1: {best_f1:.2f} | Ratio Threshold: {best_ratio_thresh}")

# Function to run ORB descriptor matching
def descriptor_ORB(directory_query, directory_bbdd, files_query, GT):
    ratio_thresholds = np.arange(0.80, 1.00, 0.05)  # Range of ratio thresholds from 0.8 to 1.00

    # Array to store metrics and predictions
    all_metrics = []

    # Iterate over different ratio_thresh values
    for ratio_thresh in ratio_thresholds:
        predicted_matches = []  # Reset predictions for each ratio_thresh

        # Iterate over each query image
        for idx, file_query in enumerate(files_query):
            # Load the query image and its ORB descriptor
            query_path = os.path.join(directory_query, file_query)
            with open(query_path, 'rb') as pkl_file:
                query_desc = pickle.load(pkl_file)

            good_matches_list = []  # List of matches for each database image
            best_match_index = None

            # Compare the query with all database images
            for file_db in os.listdir(directory_bbdd):
                if file_db.endswith('_w4.pkl') and file_db != 'relationships.pkl':
                    db_path = os.path.join(directory_bbdd, file_db)
                    with open(db_path, 'rb') as pkl_file:
                        db_desc = pickle.load(pkl_file)

                    # Compare descriptors
                    matches = find_bidirectional_matches_ORB(query_desc['desc_orb'], db_desc['desc_orb'], ratio_thresh)

                    good_matches_list.append((len(matches), extract_number_from_filename(file_db)))

            # Sort images by number of good_matches in descending order
            good_matches_list.sort(reverse=True, key=lambda x: x[0])

            # Check if the difference between top matches is significant
            if len(good_matches_list) > 1:
                first_match = good_matches_list[0]
                second_match = good_matches_list[1]
                third_match = good_matches_list[2] if len(good_matches_list) > 2 else None
                fourth_match = good_matches_list[3] if len(good_matches_list) > 3 else None

                if second_match and (first_match[0] - second_match[0]) > (first_match[0] / 3):
                    best_match_index = first_match[1]
                elif third_match and (first_match[0] - third_match[0]) > (first_match[0] / 3):
                    best_match_index = first_match[1]
                elif fourth_match and (first_match[0] - fourth_match[0]) > (first_match[0] / 3):
                    best_match_index = first_match[1]
                else:
                    # Predict -1 if differences are not significant
                    best_match_index = -1
            else:
                best_match_index = -1  

            # Predictions of the current image
            predicted_matches.append(best_match_index)
 

        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        TP = sum(1 for p, g in zip(predicted_matches, GT) if p == g)
        FP = sum(1 for p, g in zip(predicted_matches, GT) if p != g and p not in GT)
        FN = sum(1 for g, p in zip(GT, predicted_matches) if g != p and g not in predicted_matches)

        # Calculate Precision, Recall, and F1 Score manually
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        all_metrics.append((precision, recall, f1,ratio_thresh))

        # Print progess
        print(f"Processed query {idx + 1}/{len(files_query)} | Predicted: {best_match_index} | F1: {f1} | recall: {recall} | Precision: {precision}")

    best_metric = max(all_metrics, key=lambda x: x[2]) 
    best_precision, best_recall, best_f1, best_ratio_thresh = best_metric

    print(f"\nBest F1: {best_f1:.2f} | Ratio Threshold: {best_ratio_thresh}")


# Directories for the output_combinadas folder, which contains the original images without noise or ColorSpace changes, and the database
directory_query = 'output_combinadas'
directory_bbdd = '../../data/BBDD/week4'

# Get and sort the files in the query folder
files_query = sorted([f for f in os.listdir(directory_query) if f.endswith('.pkl') and f != 'gt_corresps.pkl'])

# Provided ground truth, only including single-painting photos to speed up processing
GT = [[-1], [150], [32], [161], [81], [-1], [128], [-1], [-1], [53], [-1], [12], [-1], [-1], [-1], [242], [260], [223], [-1], [127], [-1]]
GT = [item for sublist in GT for item in sublist]

# Define the range of thresholds for the ratio
descriptor_SIFT(directory_query, directory_bbdd, files_query, GT)
descriptor_ORB(directory_query, directory_bbdd, files_query, GT)