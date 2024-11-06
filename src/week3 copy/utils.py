import cv2
import numpy as np
import os
from scipy.spatial import distance
from scipy.stats import chisquare


def calculate_histogram(image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the histogram for the H, S, and V channels
    hist_h = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

    # Normalize the histograms
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    # Combine the histograms into one feature vector
    hist_combined = np.hstack([hist_h, hist_s, hist_v])

    return hist_combined


def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    return cv2.compareHist(hist1, hist2, method)


def find_most_similar_image(query_image, dataset_folder):

    query_hist = calculate_histogram(query_image)

    most_similar_image = None
    highest_similarity = -1  # Start with a very low similarity score

    # Iterate over the images in the dataset folder
    for filename in os.listdir(dataset_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(dataset_folder, filename)
            dataset_image = cv2.imread(image_path)

            if dataset_image is None:
                continue  # Skip if the image couldn't be loaded

            # Calculate the histogram for the dataset image
            dataset_hist = calculate_histogram(dataset_image)

            # Compare histograms to compute similarity
            similarity = compare_histograms(query_hist, dataset_hist)

            # Update the most similar image if the similarity score is higher
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_image = filename

    return most_similar_image, highest_similarity


def our_metric(h1, h2):
    # Input: h1, h2 (list or numpy array) - Histograms
    # Calculate the Chi-Square distance between two histograms

    h1 = np.array(h1)
    h2 = np.array(h2)
    denominator = h1 + h2
    denominator = np.where(denominator == 0, 1, denominator)  # to not divide into zero
    distance = np.sum((np.abs(h1 - h2)) / denominator)

    return distance

def euc_dist(h1, h2):
    # Calcula la distancia euclidiana entre dos histogramas usando scipy
    return distance.euclidean(h1, h2)

def L1_dist(h1, h2):
    # Calcula la distancia Manhattan (L1) entre dos histogramas usando scipy
    return distance.cityblock(h1, h2)

def X2_distance(h1, h2):
    # Input: h1, h2 (list or numpy array) - Histograms
    # Calculate the Chi-Square distance between two histograms
    h1 = h1.astype(np.float32)
    h2 = h2.astype(np.float32)
    similarity = cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR_ALT)

    return similarity

def histogram_similiarity(h1, h2):
    # Input: h1, h2 (list or numpy array) - Histogram
    # Calculate the similarity between two histograms using the intersection method
    similarity = cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)

    return similarity


def hellinger_kernel(h1, h2):
    # Input: h1, h2 (list or numpy array) - Histogram
    # Calculate the Hellinger kernel similarity between two histograms
    h1 = h1.astype(np.float32)
    h2 = h2.astype(np.float32)
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
