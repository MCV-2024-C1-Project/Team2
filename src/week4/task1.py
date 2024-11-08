import os
import cv2
import numpy as np
import pickle
from sklearn.decomposition import PCA

# def reduce_descriptor_dimensionality(descriptors, n_components=32):
#     if len(descriptors) < n_components:
#         # If there are fewer descriptors than the number of components, reduce to len(descriptors)
#         n_components = len(descriptors)
#     pca = PCA(n_components=n_components)
#     reduced_descriptors = pca.fit_transform(descriptors).astype(np.float32)
#     if reduced_descriptors.shape[1] < 32:
#         # Calculate how many zeros we need to add
#         padding_size = 32 - reduced_descriptors.shape[1]
#         # Create a zero array to complete the missing features
#         padding = np.zeros((reduced_descriptors.shape[0], padding_size), dtype=np.float32)
#         # Concatenate the zeros to the right
#         reduced_descriptors = np.hstack((reduced_descriptors, padding))
#     return reduced_descriptors

def process_directory(directory_path, flag=0):
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create()
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(directory_path, filename)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # SIFT descriptors
            kp, desc_sift = sift.detectAndCompute(image, None)
            if desc_sift is None:
                desc_sift = np.zeros((1, 128), dtype=np.float32)  # In case no keypoints are detected
            # kp, desc_sift = sift.detectAndCompute(image, None)
            # n_components = 32
            # # Verificar si los descriptores son None y manejar el caso
            # if desc_sift is None or desc_sift.shape[0] == 0:
            #     # Crear un descriptor de tamaño reducido con ceros
            #     desc_sift_reduced = np.zeros((1, n_components), dtype=np.float32)
            # else:
            #     # Reducir la dimensionalidad de los descriptores si hay suficientes descriptores
            #     if desc_sift.shape[1] != n_components:
            #         desc_sift_reduced = reduce_descriptor_dimensionality(desc_sift)
            #     else:
            #         desc_sift_reduced = desc_sift.astype(np.float32) 

            # ORB descriptors 
            kp, desc_orb = orb.detectAndCompute(image, None)
            if desc_orb is None:
                desc_orb = np.zeros((1, 32), dtype=np.uint8)  # In case no keypoints are detected

            # Combine descriptors
            histograms = {
                'desc_sift': desc_sift,
                'desc_orb': desc_orb
            }

            # Save the histograms in a pkl file
            if flag == 1:
                save_path = os.path.join(directory_path, 'week4')
                os.makedirs(save_path, exist_ok=True)
            else:
                save_path = directory_path
                
            pkl_filename = os.path.splitext(filename)[0] + '_w4.pkl'
            pkl_path = os.path.join(save_path, pkl_filename)
            print(f"Saving descriptors to: {pkl_path}")
            
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(histograms, pkl_file)

# Process both folders
directory_query1 = "filtered_cropped_qsd1_w4"

print("Current working directory:", os.getcwd())
print("Processing directory 1:")
process_directory(directory_query1, flag=0)

directory_query2 = "../../data/BBDD"
process_directory(directory_query2, flag=1)