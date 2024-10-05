import re
import pickle
import os
import utils
import numpy as np
import cv2

directory = 'qsd1_w1'
directory_bbdd = 'data/BBDD/'


for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        img_path = os.path.join(directory, filename)

        img_grey = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        hist_grey = cv2.calcHist([img_grey], [0], None, [256], [0, 256])
        hist_grey /= hist_grey.sum() # changed "hist.sum() to hist_grey.sum()"


        # RGB
        
        histograms_concatenated_rgb = []
        img_BGR = cv2.imread(img_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        hist_r = cv2.calcHist([img_RGB], [0], None, [256], [0, 256])
        hist_r /= hist_r.sum()
        hist_r=hist_r.flatten() #It is used to treat an image as a one-dimensional vector, mainly when the matrices are not the same shape. Maybe in this case we are looking at, it is not necessary because the matrices have the same shape.
        histograms_concatenated_rgb.append(hist_r)

        hist_g = cv2.calcHist([img_RGB], [1], None, [256], [0, 256])
        hist_g /= hist_g.sum()
        hist_g=hist_g.flatten()
        histograms_concatenated_rgb.append(hist_g)

        hist_b = cv2.calcHist([img_RGB], [2], None, [256], [0, 256])
        hist_b /= hist_b.sum()
        hist_b=hist_b.flatten()
        histograms_concatenated_rgb.append(hist_b)

        concatenated_hist_rgb = np.concatenate(histograms_concatenated_rgb).ravel()


        histograms = {
            'grey': hist_grey,
            'hist_RGB': concatenated_hist_rgb,
        }

        pkl_filename = os.path.splitext(filename)[0] + '.pkl'
        pkl_path = os.path.join(directory, pkl_filename)

        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(histograms, pkl_file)