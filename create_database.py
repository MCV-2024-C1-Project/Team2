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

        concatenated_hist_rgb = np.concatenate(histograms_concatenated_rgb)

        #CieLab
        histograms_concatenated_lab = []
        img_BGR = cv2.imread(img_path)
        img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
        hist_l = cv2.calcHist([img_LAB], [0], None, [256], [0, 256])
        hist_l /= hist_l.sum()
        hist_l=hist_l.flatten() 
        histograms_concatenated_lab.append(hist_l)

        hist_a = cv2.calcHist([img_LAB], [1], None, [256], [0, 256])
        hist_a /= hist_a.sum()
        hist_a=hist_a.flatten()
        histograms_concatenated_lab.append(hist_a)

        hist_b = cv2.calcHist([img_LAB], [2], None, [256], [0, 256])
        hist_b /= hist_b.sum()
        hist_b=hist_b.flatten()
        histograms_concatenated_lab.append(hist_b)

        concatenated_hist_lab = np.concatenate(histograms_concatenated_lab)

        #HSV
        img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
        histograms_concatenated_hsv = []
        
        hist_h = cv2.calcHist([img_HSV], [0], None, [256], [0, 256])
        hist_h /= hist_h.sum()
        hist_h = hist_h.flatten()  

        histograms_concatenated_hsv.append(hist_h)

        hist_s = cv2.calcHist([img_HSV], [1], None, [256], [0, 256])
        hist_s /= hist_s.sum()
        hist_s = hist_s.flatten()  
        histograms_concatenated_hsv.append(hist_s)

        hist_v = cv2.calcHist([img_HSV], [2], None, [256], [0, 256])
        hist_v /= hist_v.sum()
        hist_v = hist_v.flatten() 
        histograms_concatenated_hsv.append(hist_v)

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
directory_query = 'qsd1_w1/'

for filename in os.listdir(directory_query):
    if filename.endswith('.jpg'):
        img_path = os.path.join(directory_query, filename)

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

        concatenated_hist_rgb = np.concatenate(histograms_concatenated_rgb)

         #CieLab
        histograms_concatenated_lab = []
        img_BGR = cv2.imread(img_path)
        img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
        hist_l = cv2.calcHist([img_LAB], [0], None, [256], [0, 256])
        hist_l /= hist_l.sum()
        hist_l=hist_l.flatten() 
        histograms_concatenated_lab.append(hist_l)

        hist_a = cv2.calcHist([img_LAB], [1], None, [256], [0, 256])
        hist_a /= hist_a.sum()
        hist_a=hist_a.flatten()
        histograms_concatenated_lab.append(hist_a)

        hist_b = cv2.calcHist([img_LAB], [2], None, [256], [0, 256])
        hist_b /= hist_b.sum()
        hist_b=hist_b.flatten()
        histograms_concatenated_lab.append(hist_b)

        concatenated_hist_lab = np.concatenate(histograms_concatenated_lab)

        #HSV
        img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
        histograms_concatenated_hsv = []
        
        hist_h = cv2.calcHist([img_HSV], [0], None, [256], [0, 256])
        hist_h /= hist_h.sum()
        hist_h = hist_h.flatten()  

        histograms_concatenated_hsv.append(hist_h)

        hist_s = cv2.calcHist([img_HSV], [1], None, [256], [0, 256])
        hist_s /= hist_s.sum()
        hist_s = hist_s.flatten()  
        histograms_concatenated_hsv.append(hist_s)

        hist_v = cv2.calcHist([img_HSV], [2], None, [256], [0, 256])
        hist_v /= hist_v.sum()
        hist_v = hist_v.flatten() 
        histograms_concatenated_hsv.append(hist_v)

        concatenated_hist_hsv = np.concatenate(histograms_concatenated_hsv)

        histograms = {
            'grey': hist_grey,
            'hist_RGB': concatenated_hist_rgb,
            'hist_LAB': concatenated_hist_lab,
            'hist_HSV': concatenated_hist_hsv,
        }

        pkl_filename = os.path.splitext(filename)[0] + '.pkl'
<<<<<<< HEAD
        pkl_path = os.path.join(directory_query, pkl_filename) # changed directories 
=======
        pkl_path = os.path.join(directory_query, pkl_filename)
>>>>>>> 9c7dd5deae5ffbefe9e164a8dd66e4bae9f7ff59

        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(histograms, pkl_file)