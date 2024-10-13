import re
import os
import pickle
import utils
import numpy as np
import cv2


def spatial_pyramid_histogram(image, levels=2,resize=True,dimensions=1,hist_size=[8,8], hist_range=[0,256,0,256]):
    """
    Compute a spatial pyramid representation of histograms, with concatenation of histograms per channel.
    Level zero has 1 block. 2^0=1 so blocks 1*1=1
    Level one has 4 blocks. 2^1=2 so blocks 2*2=4
    Level two has 16 blocks. 2^2=4 so blocks 4*4=16
    """

    pyramid_hist = []
    # resize image to 256*256
    if resize==True:
        image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA) 
    h, w = image.shape[:2]  # Get the height and width of the image

    # Loop through each level in the pyramid
    for level in range(levels + 1):
        num_blocks = 2 ** level  
        block_h, block_w = h // num_blocks, w // num_blocks  # Block size

        for i in range(num_blocks):
            for j in range(num_blocks):
                # Define the block region
                block = image[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
                # Compute histograms depending on the number of channels
                block_hist = []
                if dimensions == 1:
                    # Compute 1D histogram
                    for channel in range(3):
                        hist = cv2.calcHist([block], [channel], None, hist_size, hist_range)
                        hist /= hist.sum()  
                        hist = hist.flatten()  
                        block_hist.append(hist)

                elif dimensions == 2:
                    # Compute 2D histogram 
                    lab_hist2D = cv2.calcHist([block], [0, 1], None, hist_size, hist_range)
                    # Normalize the histogram (to the range [0, 1] with NORM_MINMAX)
                    normalized = cv2.normalize(lab_hist2D, lab_hist2D, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    # Flatten the 2D histogram into a 1D vector
                    flattened_hist = normalized.flatten()
                    block_hist.append(flattened_hist)

                elif dimensions == 3:
                    # Compute 3D histogram 
                    lab_hist3D = cv2.calcHist([block], [0, 1, 2], None, hist_size, hist_range)
                    # Normalize the histogram (to the range [0, 1] with NORM_MINMAX)
                    normalized = cv2.normalize(lab_hist3D, lab_hist3D, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    # Flatten the 3D histogram into a 1D vector
                    flattened_hist = normalized.flatten()
                    block_hist.append(flattened_hist)

                # Concatenate histograms for this block
                block_hist = np.concatenate(block_hist)
                pyramid_hist.append(block_hist)

    # Concatenate all block histograms into a single feature vector
    pyramid_hist = np.concatenate(pyramid_hist)
    return pyramid_hist


def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(directory_path, filename)
            img_BGR = cv2.imread(img_path)

            # CieLab
            img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
            hist_resize_HSV_64_1D = spatial_pyramid_histogram(img_HSV, levels=2,resize=True, dimensions=1, hist_size=[64], hist_range=[0, 256])

            histograms = {
                'hist_resize_HSV_64_1D': hist_resize_HSV_64_1D,
            }

            save_path = directory_path
            pkl_filename = os.path.splitext(filename)[0] + '_w2.pkl'
            pkl_path = os.path.join(save_path, pkl_filename)
            print(pkl_path)
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(histograms, pkl_file)


# process both folders
directory_test1 = "../../datasets/qst1_w2"
print("Current working directory:", os.getcwd())
print("Processing directory 1:")
process_directory(directory_test1)

directory = '../../datasets/qst1_w2'
directory_bbdd = '../../data/BBDD/week2'


def extract_number_from_filename(filename):
    '''Function to extract the number of the image'''
    match = re.search(r'bbdd_(\d{5})_w2\.pkl', filename)
    if match:
        return int(match.group(1))


def extract_number_from_filename_qsd1_w1(filename):
    '''Function to extract the number of the image'''
    match = re.search(r'(\d{5})_w2\.pkl', filename)
    if match:
        return int(match.group(1))


# Method 1
min_distance = float('inf')
list_results_k_10 = []

files = os.listdir(directory)
files_sorted = sorted(files)

for file_compare_image in files_sorted:
    # catch the first image then the second and so on of the qsd1_w1

    if file_compare_image.endswith('_w2.pkl') and file_compare_image != 'gt_corresps.pkl':
        pkl_grey_path = os.path.join(directory, file_compare_image)
        with open(pkl_grey_path, 'rb') as pkl_file:
            histograms_first = pickle.load(pkl_file)

        distances = []
        index_qsd1_w1 = extract_number_from_filename_qsd1_w1(file_compare_image)

        for filename in os.listdir(directory_bbdd):
            if filename.endswith('_w2.pkl') and filename != 'relationships.pkl':
                pkl_path = os.path.join(directory_bbdd, filename)

                with open(pkl_path, 'rb') as pkl_file:
                    histograms = pickle.load(pkl_file)

                histogram_first_grey = histograms_first['hist_resize_HSV_64_1D']
                histogram_grey = histograms['hist_resize_HSV_64_1D']
                # # Try all 4 loss functions: euc_dist, L1_dist, X2_distance, hellinger_kernel, histogram_similiarity
                distance = utils.our_metric(histogram_first_grey, histogram_grey)

                index = extract_number_from_filename(filename)

                distances.append((distance, index))

        # Sort the distances and select the top k results
        distances.sort(key=lambda x: x[0])  # Sort by distance (lowest first)
        top_k_10_results = [index for _, index in distances[:10]]

        # when the loop for the first query image is finish we save the index in a list
        list_results_k_10.append(top_k_10_results)

list_results_k_10 = [result for result in list_results_k_10]

# Para los resultados aplanados
predicted_flattened_k_10 = [p for p in list_results_k_10]

# To ensure that they are integers
list_results_k_10_enteros = [[int(x) for x in sublist] for sublist in list_results_k_10]

print(list_results_k_10_enteros)

# store results k=10
with open('../../results/week2/QST1/method1/result.pkl', 'wb') as pkl_file:
    pickle.dump(list_results_k_10_enteros, pkl_file)
