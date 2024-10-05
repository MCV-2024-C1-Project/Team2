import numpy as np
import cv2
import os
import pickle

import utils

def load_and_print_pkl(pkl_file_path):
    # Input: pkl_file_path (string) - Path to the .pkl file
    # Load and print the contents of a pickle (.pkl) file
    with open(pkl_file_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)      
        print("Content of the pickle file:")
        print(data)

load_and_print_pkl('data/BBDD/bbdd_00003.pkl')
