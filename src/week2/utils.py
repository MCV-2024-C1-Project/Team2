import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------
# Author: Agustina Ghelfi, Grigor Grigoryan, Philip Zetterberg, Vincent Heuer
# Date: 03.10.2024
#
# Description:
# This library contains useful functions for use in CV problems. The functions are used to solve the tasks of the MCV24 C1 class and will be expanded as needed during the class.
#
# ---------------------------------------------------------------------------------


def hist_plot_grey(img):
    # Input: img (numpy array) - BGR image
    # Convert image to greyscale and calculate the histogram (normalized)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img_grey], [0], None, [256], [0, 256])
    hist /= hist.sum()

    # Plot the greyscale histogram
    plt.plot(hist, colour='black')
    plt.title('Histogram of Greyscale Value')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    plt.show()


def hist_plot_RGB(img):
    # Input: img (numpy array) - BGR image
    # Split the image into its B, G, R channels and calculate histograms for each
    B, G, R = cv2.split(img)
    hist_B = cv2.calcHist([B], [0], None, [256], [0, 256])
    hist_G = cv2.calcHist([G], [0], None, [256], [0, 256])
    hist_R = cv2.calcHist([R], [0], None, [256], [0, 256])

    # Plot histograms for Red, Green, and Blue channels
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(hist_R, color='red')
    plt.title('Histogram of Red Channel')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])

    plt.subplot(1, 3, 2)
    plt.plot(hist_G, color='green')
    plt.title('Histogram of Green Channel')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])

    plt.subplot(1, 3, 3)
    plt.plot(hist_B, color='blue')
    plt.title('Histogram of Blue Channel')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])

    plt.show()


def euc_dist(h1, h2):
    # Input: h1, h2 (list or numpy array) - Histograms 
    # Calculate the Euclidean distance between two histograms  

    h1 = np.array(h1)
    h2 = np.array(h2)
    distance = np.sqrt(np.sum((h1 - h2) ** 2))

    return distance


def L1_dist(h1, h2):
    # Input: h1, h2 (list or numpy array) - Histograms
    # Calculate the L1 (Manhattan) distance between two histograms

    h1 = np.array(h1)
    h2 = np.array(h2)
    distance = np.sum(np.abs(h1 - h2))

    return distance


def X2_distance(h1, h2):
    # Input: h1, h2 (list or numpy array) - Histograms
    # Calculate the Chi-Square distance between two histograms

    similarity = cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR_ALT)

    return similarity


def our_metric(h1, h2):
    # Input: h1, h2 (list or numpy array) - Histograms
    # Calculate the Chi-Square distance between two histograms

    h1 = np.array(h1)
    h2 = np.array(h2)
    denominator = h1 + h2
    denominator = np.where(denominator == 0, 1, denominator)  # to not divide into zero
    distance = np.sum((np.abs(h1 - h2)) / denominator)

    return distance


def histogram_similiarity(h1, h2):
    # Input: h1, h2 (list or numpy array) - Histogram
    # Calculate the similarity between two histograms using the intersection method

    similarity = cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)

    return similarity


def hellinger_kernel(h1, h2):
    # Input: h1, h2 (list or numpy array) - Histogram
    # Calculate the Hellinger kernel similarity between two histograms

    similarity = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

    return similarity


def load_and_print_pkl(pkl_file_path):
    # Input: pkl_file_path (string) - Path to the .pkl file
    # Load and print the contents of a pickle (.pkl) file
    with open(pkl_file_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)      
        print("Content of the pickle file:")
        print(data)


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def compute_precision_recall_f1(actual, predicted):
    """
    Compute precision, recall, and F1 score given the actual (ground truth) and predicted labels.
    
    Parameters:
    - actual: list or array-like, the ground truth labels.
    - predicted: list or array-like, the predicted labels.
    
    Returns:
    - precision: float, the precision score.
    - recall: float, the recall score.
    - f1: float, the F1 score.
    """

    # Convert inputs to sets to remove duplicates
    actual_set = set(actual)
    predicted_set = set(predicted)
    
    # Compute true positives: items correctly predicted
    true_positives = len(actual_set.intersection(predicted_set))
    
    # Compute precision: proportion of correct predictions over all predictions
    precision = true_positives / len(predicted_set) if len(predicted_set) > 0 else 0
    
    # Compute recall: proportion of correct predictions over all actual ground truths
    recall = true_positives / len(actual_set) if len(actual_set) > 0 else 0

    # Compute F1 score: harmonic mean of precision and recall
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1