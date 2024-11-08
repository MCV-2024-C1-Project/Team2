import os
import re
import pickle
import numpy as np
import pandas as pd
from task1_2 import process_and_match


def extract_number_from_filename(filename):
    '''Extracts the image number from the filename for sorting and identification'''
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))


# Custom mapk function - unchanged
def custom_mapk(actual, predicted, k=10):
    def apk_single(actual, predicted, k):
        if len(predicted) > k:
            predicted = predicted[:k]
        return 1.0 if actual[0] in predicted else 0.0

    def apk_double(actual, predicted, k):
        if isinstance(predicted[0], list):
            if len(predicted[0]) > k:
                predicted = [pred[:k] for pred in predicted]
            if actual[0] in predicted[0] and actual[1] in predicted[1]:
                return 1.0
            elif actual[0] in predicted[0] or actual[1] in predicted[1]:
                return 0.5
            else:
                return 0.0
        else:
            if actual[0] == predicted[0] and actual[1] == predicted[1]:
                return 1.0
            elif actual[0] == predicted[0] or actual[1] == predicted[1]:
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


# Function to get matches for an image file
def get_image_matches(query_path, bbdd_directory, method="ORB", metric="L2"):
    matches = []
    for bbdd_filename in sorted(os.listdir(bbdd_directory)):
        if bbdd_filename.endswith('.jpg'):
            match_score = process_and_match(
                query_path, os.path.join(bbdd_directory, bbdd_filename),
                method=method, metric=metric
            )
            image_number = extract_number_from_filename(bbdd_filename)
            matches.append((match_score, image_number))
    matches.sort(key=lambda x: x[0])
    return [match[1] for match in matches[:10]]  # Top 10 matches


# Process matches for each query image, handling one or two masks
def process_matches_with_masks(directory, bbdd_directory):
    all_matches = []
    for filename_query in sorted(os.listdir(directory)):
        if filename_query.endswith('.jpg'):
            # Process first mask
            mask_1_matches = get_image_matches(
                os.path.join(directory, filename_query), bbdd_directory
            )
            contour2_file = filename_query.replace('_contour1', '_contour2')

            # Check for second mask and process it if available
            if os.path.exists(os.path.join(directory, contour2_file)):
                mask_2_matches = get_image_matches(
                    os.path.join(directory, contour2_file), bbdd_directory
                )
                all_matches.append([mask_1_matches, mask_2_matches])
            else:
                all_matches.append([mask_1_matches])  # Single mask case
    return all_matches


# Calculate mAP@k for multiple k values
def calculate_mapk(matches, ground_truth, k_values=[1, 5, 10]):
    results = {}
    for k in k_values:
        results[f'map@{k}'] = custom_mapk(ground_truth, matches, k)
    return results


# Start
directory = 'filtered_cropped_qst1_w4'  # Directory containing the filtered images
directory_bbdd = '../../data/BBDD/week4'  # Directory containing the database of images

# Run the matching process and evaluate
all_matches = process_matches_with_masks(directory, directory_bbdd)

# Print the results
for i in range(30):
    print(i, all_matches[i])

print(all_matches)

# Store results for k=10
output_path = '../../results/week4/QST1/method1/result.pkl'
with open(output_path, 'wb') as pkl_file:
    pickle.dump(all_matches, pkl_file)

print(f"Results saved to {output_path}")


# Load Ground Truth only if self made
# with open(os.path.join(directory, 'gt_corresps.pkl'), 'rb') as f:
#    ground_truth = pickle.load(f)

# mapk_results = calculate_mapk(all_matches, ground_truth, k_values=[1, 5, 10])

# Save results to CSV
# results_df = pd.DataFrame([mapk_results])
# results_df.to_csv("keypoint_mapk_results.csv", index=False)

# print("Results:", mapk_results)
