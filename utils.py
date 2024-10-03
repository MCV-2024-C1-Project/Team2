import numpy as np

# ---------------------------------------------------------------------------------
# Author: Agustina Ghelfi, Grigor Grigoryan, Philip Zetterberg, Vincent Heuer
# Date: 03.10.2024
#
# Description:
# This Python script contains useful functions for use in CV problems. The functions are used to solve the tasks of the MCV24 C1 class and will be expanded as needed during the class.
#
# ---------------------------------------------------------------------------------

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

    distance = np.sum((np.sqrt(h1 - h2) ** 2) / (h1 + h2))

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

def load_and_print_pkl(pkl_file_path):
    with open(pkl_file_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)      
        print("Content of the pickle file:")
        print(data)
