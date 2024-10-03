import os
import pickle
import cv2

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
        # RGB
        img_RGB=cv2.imread(img_path) 
        B, G, R = cv2.split(img_RGB) # cv2.split returns the three RGB channels of the image in the order blue, green and then red

        hist_B = cv2.calcHist([B], [0], None, [256], [0, 256])
        hist_B /= hist_R.sum()

        hist_G = cv2.calcHist([G], [0], None, [256], [0, 256])
        hist_G /= hist_G.sum()

        hist_R = cv2.calcHist([R], [0], None, [256], [0, 256])
        hist_R /= hist_B.sum()

        # Save descriptors as .pkl file
        histograms = {
            'grey': hist_grey,
            'hist_B': hist_B,
            'hist_G': hist_G,
            'hist_R': hist_R,
        }

        pkl_filename = os.path.splitext(filename)[0] + '.pkl'
        pkl_path = os.path.join(directory, pkl_filename)

        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(histograms, pkl_file)
        
