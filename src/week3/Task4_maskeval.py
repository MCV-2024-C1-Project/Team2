import os
import numpy as np
import pickle
import cv2


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

# Process all .pkl files in the folder and accumulate Precision, Recall, and F1-score
def process_folder_and_evaluate_pkl(pkl_folder):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_files = 0

    # Iterate over all .pkl files in the folder
    for filename in os.listdir(pkl_folder):
        if filename.endswith('_mask_s_contour1.png'):
            contour2_filename = filename.replace('_mask_s_contour1.png', '_mask_s_contour2.png')

            if contour2_filename in os.listdir(pkl_folder):
                # load two masks and join them
                image_path1 = os.path.join(pkl_folder, filename)
                image_path2 = os.path.join(pkl_folder, contour2_filename)
                mask1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
                mask2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
                joined_mask = np.maximum(mask1, mask2)
                cv2.imshow('image', joined_mask); cv2.waitKey(0); cv2.destroyAllWindows()

                # Load the corresponding ground truth mask (as .png)
                corresponding_image_path = os.path.join(pkl_folder, filename.replace('_mask_s_contour1.png', '.png'))
                ground_truth_mask = cv2.imread(corresponding_image_path, cv2.IMREAD_GRAYSCALE)
                if ground_truth_mask is None:
                    print(f"Error loading ground truth for {filename}, skipping.")
                    continue

                # Evaluate the generated mask against the ground truth mask using precision, recall, and F1-score
                precision, recall, f1_score = evaluate_mask_precision_recall_f1(joined_mask, ground_truth_mask)

                # Print evaluation results for the file
                print(f"Results for {filename}:")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-score: {f1_score:.4f}")

                # Accumulate the metrics for overall performance
                total_precision += precision
                total_recall += recall
                total_f1 += f1_score
                num_files += 1

            else:
           
                #pkl_path = os.path.join(pkl_folder, filename)
                corresponding_image_path = os.path.join(pkl_folder, filename.replace('_mask_s_contour1.png', '.png'))
                image_path = os.path.join(pkl_folder, filename)
                # Load the .pkl file (which contains the masks)
                #with open(pkl_path, 'rb') as file:
                    #masks = pickle.load(file)
                masks = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                cv2.imshow('image', masks); cv2.waitKey(0); cv2.destroyAllWindows()
                # Choose the mask to evaluate based on the parameter
                #if mask_type not in masks:
                    #print(f"Error: mask type '{mask_type}' not found in {filename}. Skipping.")
                    #continue

                generated_mask = masks    #[mask_type]

                # Load the corresponding ground truth mask (as .png)
                ground_truth_mask = cv2.imread(corresponding_image_path, cv2.IMREAD_GRAYSCALE)

                if ground_truth_mask is None:
                    print(f"Error loading ground truth for {filename}, skipping.")
                    continue

                # Evaluate the generated mask against the ground truth mask using precision, recall, and F1-score
                precision, recall, f1_score = evaluate_mask_precision_recall_f1(generated_mask, ground_truth_mask)

                # Print evaluation results for the file
                print(f"Results for {filename}:")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-score: {f1_score:.4f}")

                # Accumulate the metrics for overall performance
                total_precision += precision
                total_recall += recall
                total_f1 += f1_score
                num_files += 1

    # Compute and print the average performance over all files
    if num_files > 0:
        avg_precision = total_precision / num_files
        avg_recall = total_recall / num_files
        avg_f1 = total_f1 / num_files

        print("\nOverall Performance:")
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Average F1-score: {avg_f1:.4f}")
    else:
        print("No valid files were processed.")


directory = 'datasets/qsd2_w3'

process_folder_and_evaluate_pkl(directory)