import numpy as np

# ---------------------------------------------------------------------------------
# Author: Agustina Ghelfi, Grigor Grigoryan, Philip Zetterberg, Vincent Heuer
# Date: 03.10.2024
#
# Description:
# This Python script contains useful functions for use in CV problems. The functions are used to solve the tasks of the MCV24 C1 class and will be expanded as needed during the class.
#
# ---------------------------------------------------------------------------------

def hist_plot_grey(img):
    # Input: img (numpy array) - BGR image
    # Convert image to greyscale and calculate the histogram (normalized)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
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
    # Input: h1, h2 (list or numpy array) - Histograms with 256 bins
    # Calculate the Euclidean distance between two histograms
    if len(h1) != 256 or len(h2) != 256:
        raise ValueError("Both histograms must have a length of 256")    
    
    h1 = np.array(h1)
    h2 = np.array(h2)
    distance = np.sqrt(np.sum((h1 - h2) ** 2))

    return distance


def L1_dist(h1, h2):
    # Input: h1, h2 (list or numpy array) - Histograms with 256 bins
    # Calculate the L1 (Manhattan) distance between two histograms
    if len(h1) != 256 or len(h2) != 256:
        raise ValueError("Both histograms must have a length of 256")
    
    h1 = np.array(h1)
    h2 = np.array(h2)
    distance = np.sum(np.abs(h1 - h2))
    
    return distance


def X2_distance(h1, h2):
    # Input: h1, h2 (list or numpy array) - Histograms with 256 bins
    # Calculate the Chi-Square distance between two histograms
    if len(h1) != 256 or len(h2) != 256:
        raise ValueError("Both histograms must have a length of 256")

    h1 = np.array(h1)
    h2 = np.array(h2)
    distance = np.sum((np.sqrt(h1 - h2) ** 2) / (h1 + h2))

    return distance


def histogram_similiarity(h1, h2):
    # Input: h1, h2 (list or numpy array) - Histograms with 256 bins
    # Calculate the similarity between two histograms using the intersection method
    if len(h1) != 256 or len(h2) != 256:
        raise ValueError("Both histograms must have a length of 256")

    h1 = np.array(h1)
    h2 = np.array(h2)
    similiarity = np.sum(np.minimum(h1, h2))

    return similiarity


def hellinger_kernel(h1, h2):
    # Input: h1, h2 (list or numpy array) - Histograms with 256 bins
    # Calculate the Hellinger kernel similarity between two histograms
    if len(h1) != 256 or len(h2) != 256:
        raise ValueError("Both histograms must have a length of 256")
        
    h1 = np.array(h1)
    h2 = np.array(h2)
    similiarity = np.sum(np.sqrt(h1 * h2))

    return similiarity


def load_and_print_pkl(pkl_file_path):
    # Input: pkl_file_path (string) - Path to the .pkl file
    # Load and print the contents of a pickle (.pkl) file
    with open(pkl_file_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)      
        print("Content of the pickle file:")
        print(data)

