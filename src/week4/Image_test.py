import cv2
import os
import numpy as np

directory = 'datasets/qsdq_w4/'

GT_path = 'datasets/qsd1_w4/00004.png'

mask_path = 'datasets/qsd1_w4/00004_seg_contour1.png'

GT = cv2.imread(GT_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if GT is None:
    print(f"Error loading ground truth for {GT_path}, skipping.")
 
if mask is None:
    print(f"Error loading mask for {mask_path}, skipping.")
    

cv2.imshow('GT', GT), cv2.waitKey(0), cv2.imshow('mask', mask), cv2.waitKey(0), cv2.destroyAllWindows()

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

precision, recall, f1_score = evaluate_mask_precision_recall_f1(mask, GT)
print(f"\nPrecision: {precision}, Recall: {recall}, F1-score: {f1_score}")