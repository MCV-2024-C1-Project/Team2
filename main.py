

import cv2
import matplotlib.pyplot as plt

# Local application/library specific imports

import utils

# ---------------------------------------------------------------------------------
# Author: Agustina Ghelfi, Grigor Grigoryan, Philip Zetterberg, Vincent Heuer
# Date: 03.10.2024
# Version: 1.0
# 
# Version 1.0 tackles the tasks of the first week
#
# Description:
# This Python script proposes a solution to building a query for museum paintings using CV techniques such as 1D histogram comparison and more.
#
# In the future this mainscript is expanded to enhance the system's capability and to tackle the upcoming challenges.
# ---------------------------------------------------------------------------------

#----------<main>-----------
#----------<task 1>---------

directory = 'data/BBDD'

for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        # Define the corresponding pickle file path
        pkl_filename = os.path.splitext(filename)[0] + '.pkl'
        pkl_path = os.path.join(directory, pkl_filename)

        # Check if the pickle file already exists
        if os.path.exists(pkl_path):
            continue  # Skip to the next image

        # Proceed with image processing if no pickle file is found
        img_path = os.path.join(directory, filename)

        # Grayscale Histogram
        img_grey = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        hist_grey = cv2.calcHist([img_grey], [0], None, [256], [0, 256])
        hist_grey /= hist_grey.sum()  # Normalize the histogram

        # RGB Histograms
        img_RGB = cv2.imread(img_path)
        B, G, R = cv2.split(img_RGB)

        hist_B = cv2.calcHist([B], [0], None, [256], [0, 256])
        hist_B /= hist_B.sum()  # Normalize the histogram

        hist_G = cv2.calcHist([G], [0], None, [256], [0, 256])
        hist_G /= hist_G.sum()  # Normalize the histogram

        hist_R = cv2.calcHist([R], [0], None, [256], [0, 256])
        hist_R /= hist_R.sum()  # Normalize the histogram

        # Store histograms in a dictionary
        histograms = {
            'grey': hist_grey,
            'hist_B': hist_B,
            'hist_G': hist_G,
            'hist_R': hist_R,
        }

        # Save the histograms to a pickle file
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(histograms, pkl_file)

        print(f"Processed and saved histograms for {filename}")

print("Task 1 complete.") 

#----------<task 2 & 3>---------
