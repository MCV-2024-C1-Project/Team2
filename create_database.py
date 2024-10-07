import os
import pickle
import cv2
import numpy as np

# ---------------------------------------------------------------------------------
# Author: Agustina Ghelfi, Grigor Grigoryan, Philip Zetterberg, Vincent Heuer
# Date: 03.10.2024
#
#
# Description:
# This Python script was used to create a database from the training data. This loads the images and extracts descriptors and ultimately saves a list of lists for each image in a .pkl file.
#
# In the future this script might be expanded to generate more descriptors.
# ---------------------------------------------------------------------------------

directory = 'data/BBDD/' # Specify the directory of the database and the images
# Loop through the data and perform operations
for filename in os.listdir(directory): 
    if filename.endswith('.jpg'):
        img_path = os.path.join(directory, filename)

        # Extract descriptors from image greyscale
        img_grey = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        hist_grey = cv2.calcHist([img_grey], [0], None, [256], [0, 256])
        hist_grey /= hist_grey.sum() # changed "hist.sum() to hist_grey.sum()"

        # Extract descriptors from RGB channels
        histograms_concatenated_rgb = []
        img_BGR = cv2.imread(img_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        for channel in range(3):
            hist = cv2.calcHist([img_RGB], [channel], None, [256], [0, 256])
            hist /= hist.sum()  
            hist = hist.flatten() 
            histograms_concatenated_rgb.append(hist)

        concatenated_hist_rgb = np.concatenate(histograms_concatenated_rgb)

        #CieLab
        histograms_concatenated_lab = []
        img_BGR = cv2.imread(img_path)
        img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
        for channel in range(3):
            hist = cv2.calcHist([img_LAB], [channel], None, [256], [0, 256])
            hist /= hist.sum()  
            hist = hist.flatten()  
            histograms_concatenated_lab.append(hist)

        concatenated_hist_lab = np.concatenate(histograms_concatenated_lab)

        #HSV
        img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
        histograms_concatenated_hsv = []
        for channel in range(3):
            hist = cv2.calcHist([img_HSV], [channel], None, [256], [0, 256])
            hist /= hist.sum()  
            hist = hist.flatten()  
            histograms_concatenated_hsv.append(hist)

        concatenated_hist_hsv = np.concatenate(histograms_concatenated_hsv)

        histograms = {
            'grey': hist_grey,
            'hist_RGB': concatenated_hist_rgb,
            'hist_LAB': concatenated_hist_lab,
            'hist_HSV': concatenated_hist_hsv,
        }

        pkl_filename = os.path.splitext(filename)[0] + '.pkl'
        pkl_path = os.path.join(directory, pkl_filename)

        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(histograms, pkl_file)
            
print('Finish the data folder proccessing')

directory_query = 'datasets/qsd1_w1/'

for filename in os.listdir(directory_query):
    if filename.endswith('.jpg'):
        img_path = os.path.join(directory_query, filename)

        # Extract descriptors from image greyscale
        img_grey = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        hist_grey = cv2.calcHist([img_grey], [0], None, [256], [0, 256])
        hist_grey /= hist_grey.sum() # changed "hist.sum() to hist_grey.sum()"

        # Extract descriptors from RGB channels
        histograms_concatenated_rgb = []
        img_BGR = cv2.imread(img_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        for channel in range(3):
            hist = cv2.calcHist([img_RGB], [channel], None, [256], [0, 256])
            hist /= hist.sum()  
            hist = hist.flatten() 
            histograms_concatenated_rgb.append(hist)

        concatenated_hist_rgb = np.concatenate(histograms_concatenated_rgb)

        #CieLab
        histograms_concatenated_lab = []
        img_BGR = cv2.imread(img_path)
        img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
        for channel in range(3):
            hist = cv2.calcHist([img_LAB], [channel], None, [256], [0, 256])
            hist /= hist.sum()  
            hist = hist.flatten()  
            histograms_concatenated_lab.append(hist)

        concatenated_hist_lab = np.concatenate(histograms_concatenated_lab)

        #HSV
        img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
        histograms_concatenated_hsv = []
        for channel in range(3):
            hist = cv2.calcHist([img_HSV], [channel], None, [256], [0, 256])
            hist /= hist.sum()  
            hist = hist.flatten()  
            histograms_concatenated_hsv.append(hist)

        concatenated_hist_hsv = np.concatenate(histograms_concatenated_hsv)

        histograms = {
            'grey': hist_grey,
            'hist_RGB': concatenated_hist_rgb,
            'hist_LAB': concatenated_hist_lab,
            'hist_HSV': concatenated_hist_hsv,
        }

        pkl_filename = os.path.splitext(filename)[0] + '.pkl'
        pkl_path = os.path.join(directory_query, pkl_filename)

        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(histograms, pkl_file)

print('Finish the qsd1_w1 folder proccessing')


directory_query = 'datasets/qst1_w1/'

for filename in os.listdir(directory_query):
    if filename.endswith('.jpg'):
        img_path = os.path.join(directory_query, filename)

        # Extract descriptors from image greyscale
        img_grey = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        hist_grey = cv2.calcHist([img_grey], [0], None, [256], [0, 256])
        hist_grey /= hist_grey.sum() # changed "hist.sum() to hist_grey.sum()"

        # Extract descriptors from RGB channels
        histograms_concatenated_rgb = []
        img_BGR = cv2.imread(img_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        for channel in range(3):
            hist = cv2.calcHist([img_RGB], [channel], None, [256], [0, 256])
            hist /= hist.sum()  
            hist = hist.flatten() 
            histograms_concatenated_rgb.append(hist)

        concatenated_hist_rgb = np.concatenate(histograms_concatenated_rgb)

        #CieLab
        histograms_concatenated_lab = []
        img_BGR = cv2.imread(img_path)
        img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
        for channel in range(3):
            hist = cv2.calcHist([img_LAB], [channel], None, [256], [0, 256])
            hist /= hist.sum()  
            hist = hist.flatten()  
            histograms_concatenated_lab.append(hist)

        concatenated_hist_lab = np.concatenate(histograms_concatenated_lab)

        #HSV
        img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
        histograms_concatenated_hsv = []
        for channel in range(3):
            hist = cv2.calcHist([img_HSV], [channel], None, [256], [0, 256])
            hist /= hist.sum()  
            hist = hist.flatten()  
            histograms_concatenated_hsv.append(hist)

        concatenated_hist_hsv = np.concatenate(histograms_concatenated_hsv)

        histograms = {
            'grey': hist_grey,
            'hist_RGB': concatenated_hist_rgb,
            'hist_LAB': concatenated_hist_lab,
            'hist_HSV': concatenated_hist_hsv,
        }

        pkl_filename = os.path.splitext(filename)[0] + '.pkl'
        pkl_path = os.path.join(directory_query, pkl_filename)

        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(histograms, pkl_file)

print('Finish the qst1_w1 folder proccessing')
