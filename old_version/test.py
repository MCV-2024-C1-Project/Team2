import numpy as np
import os
import pickle
import re


def euc_dist(h1, h2):

    if len(h1) != 256 or len(h2) != 256:
        raise ValueError("Both histograms must have a length of 256")

    h1 = np.array(h1)
    h2 = np.array(h2)

    distance = np.sqrt(np.sum((h1 - h2) ** 2))

    return distance


def L1_dist(h1, h2):

    if len(h1) != 256 or len(h2) != 256:
        raise ValueError("Both histograms must have a length of 256")

    h1 = np.array(h1)
    h2 = np.array(h2)

    distance = np.sum(np.abs(h1 - h2))

    return distance


def X2_distance(h1, h2):

    if len(h1) != 256 or len(h2) != 256:
        raise ValueError("Both histograms must have a length of 256")

    h1 = np.array(h1)
    h2 = np.array(h2)

    distance = np.sum(((h1 - h2) ** 2) / (h1 + h2))

    return distance


def histogram_similiarity(h1, h2):

    if len(h1) != 256 or len(h2) != 256:
        raise ValueError("Both histograms must have a length of 256")

    h1 = np.array(h1)
    h2 = np.array(h2)

    similiarity = np.sum(np.minimum(h1, h2))

    return similiarity


def hellinger_kernel(h1, h2):

    if len(h1) != 256 or len(h2) != 256:
        raise ValueError("Both histograms must have a length of 256")

    h1 = np.array(h1)
    h2 = np.array(h2)

    similiarity = np.sum(np.sqrt(h1*h2))

    return similiarity


test_directory = 'qsd1_w1'
museum_directory = 'data/BBDD'


for filename_test in os.listdir(test_directory):
    if filename_test.endswith('.pkl') and filename_test != 'gt_corresps.pkl':

        pkl_path_test = os.path.join(test_directory, filename_test)

        with open(pkl_path_test, 'rb') as pkl_file:
            histograms_test = pickle.load(pkl_file)

        match = re.search(r'\d+', filename_test)
        integer_test = int(match.group())

        for filename_museum in os.listdir(museum_directory):
            if filename_test.endswith('.pkl') and filename_test != 'relationships.pkl':

                pkl_path_museum = os.path.join(test_directory, filename_test)

                with open(pkl_path_museum, 'rb') as pkl_file:
                    histograms_museum = pickle.load(pkl_file)

                match = re.search(r'\d+', filename_test)
                integer_test = int(match.group())

                euc_dist(histograms_test['grey'], histograms_museum['grey'])
                L1_dist(histograms_test['grey'], histograms_museum['grey'])
                X2_distance(histograms_test['grey'], histograms_museum['grey'])
                histogram_similiarity(histograms_test['grey'], histograms_museum['grey'])
                hellinger_kernel(histograms_test['grey'], histograms_museum['grey'])
