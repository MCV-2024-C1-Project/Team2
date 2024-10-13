import cv2
import numpy as np
import os


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
