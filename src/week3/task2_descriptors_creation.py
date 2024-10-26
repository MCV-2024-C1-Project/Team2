import numpy as np
import os
import pickle
import cv2
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct
import pywt

def lbp_descriptor_by_block_size(image, num_points=8, radius=1, block_size=8):
    fixed_size = (256, 256)
    resized_image = cv2.resize(image, fixed_size)
    h, w = resized_image.shape[:2]
    lbp_blocks = []

    # Loop over the blocks defined by block_size
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Define the block region
            block = resized_image[i:i + block_size, j:j + block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size: 
                if len(image.shape) == 2:  # Grayscale image
                    # Apply LBP to the block
                    lbp = local_binary_pattern(block, num_points, radius, method="uniform")
                    lbp_uint8 = np.uint8(lbp)
                    # Compute histogram of the LBP result
                    hist, _ = np.histogram(lbp_uint8.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
                    hist = hist / np.sum(hist)
                    lbp_blocks.append(hist)  # Store the histogram for the block
                else:  # Color image (RGB)
                    for channel in range(3):
                        # Apply LBP to the corresponding channel
                        lbp = local_binary_pattern(block[:, :, channel], num_points, radius, method="uniform")
                        lbp_uint8 = np.uint8(lbp)
                        # Compute histogram of the LBP result
                        hist, _ = np.histogram(lbp_uint8.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
                        hist = hist / np.sum(hist)
                        lbp_blocks.append(hist)  

    # Concatenate all block histograms into a single feature vector
    feature_vector = np.concatenate(lbp_blocks)
    return feature_vector 
 
def wavelet_descriptor_by_block_size(image, wavelet='haar', block_size=8):
    fixed_size = (256, 256)
    resized_image = cv2.resize(image, fixed_size)
    h, w = resized_image.shape[:2]
    wavelet_blocks = []

    # Loop over the blocks defined by block_size
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Define the block region
            block = resized_image[i:i + block_size, j:j + block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                if len(image.shape) == 2:  # Grayscale image
                    # Apply DWT to the block
                    cA, (cH, cV, cD) = pywt.dwt2(block, wavelet)
                    # Concatenate the coefficients of the subbands into a single vector
                    wavelet_descriptor = np.concatenate([cA.ravel(), cH.ravel(), cV.ravel(), cD.ravel()])
                    wavelet_blocks.append(wavelet_descriptor)
                else:  # Color image (RGB)
                    for channel in range(3):
                        # Apply DWT to each channel of the block
                        cA, (cH, cV, cD) = pywt.dwt2(block[:, :, channel], wavelet)
                        # Concatenate the coefficients of the subbands
                        wavelet_descriptor = np.concatenate([cA.ravel(), cH.ravel(), cV.ravel(), cD.ravel()])
                        wavelet_blocks.append(wavelet_descriptor)

    # Concatenate all block descriptors into a single feature vector
    feature_vector = np.concatenate(wavelet_blocks)
    return feature_vector

def multiscale_lbp_descriptor_color(image, radius_values, num_points=8, block_size=8):
    fixed_size = (256, 256)
    resized_image = cv2.resize(image, fixed_size)
    h, w, _ = resized_image.shape
    lbp_descriptors = []

    # Split the resized image into color channels
    channels = cv2.split(resized_image)

    # For each color channel (R, G, B)
    for channel in channels:
        channel_descriptors = []
        
        # Divide the image into blocks
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = channel[i:i + block_size, j:j + block_size]
                block_descriptors = []
                
                # Calculate LBP at multiple scales for the block
                for radius in radius_values:
                    # Apply LBP
                    lbp = local_binary_pattern(block, num_points, radius, method="uniform")
                    lbp_uint8 = np.uint8(lbp)
                    hist, _ = np.histogram(lbp_uint8.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
                    hist = hist / np.sum(hist)  
                    block_descriptors.append(hist)
                
                # Concatenate descriptors for each scale in the block
                channel_descriptors.append(np.concatenate(block_descriptors))
        
        # Concatenate LBP descriptors for this channel
        lbp_descriptors.append(np.concatenate(channel_descriptors))
    
    # Concatenate descriptors from all three channels into a single feature vector
    feature_vector = np.concatenate(lbp_descriptors)
    return feature_vector


def zigzag_scan(matrix):
    rows, cols = matrix.shape
    result = []
    for i in range(rows + cols - 1):
        if i % 2 == 0:
            # Even index: traverse the diagonal from bottom to top
            for j in range(min(i, rows - 1), max(-1, i - cols), -1):
                result.append(matrix[j][i - j])
        else:
            # Odd index: traverse the diagonal from top to bottom
            for j in range(max(0, i - cols + 1), min(i + 1, rows)):
                result.append(matrix[j][i - j])
    return np.array(result) 

def dct_descriptor_color(image, block_size=8, num_coefs=10):
    fixed_size = (256, 256)
    resized_image = cv2.resize(image, fixed_size)
    channels = cv2.split(resized_image)
    h, w = resized_image.shape[:2]
    dct_blocks = []
    
    for channel in channels:
        # Iterate over blocks in the channel
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = channel[i:i + block_size, j:j + block_size]
                if block.shape == (block_size, block_size):
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')  # Perform DCT in two directions
                    # Perform zigzag scanning on the DCT block
                    zz = zigzag_scan(dct_block)
                    # Keep the first N coefficients
                    dct_blocks.append(zz[:num_coefs])
    
    # Concatenate all blocks from the three channels into a single feature vector
    feature_vector = np.concatenate(dct_blocks)
    return feature_vector  

def process_directory(directory_path,flag=0):
    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(directory_path, filename)

            # Extract descriptors from YCbCr channels
            img_BGR = cv2.imread(img_path)
            img_ycbcr = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
            #LBP for ycbcr
            #hist_LBP_ycbcr_n8_1D=lbp_descriptor_by_block_size(img_ycbcr,num_points=8,radius=1,block_size=8)
            hist_LBP_ycbcr_n8_r2_1D=lbp_descriptor_by_block_size(img_ycbcr,num_points=8,radius=2,block_size=8)
            hist_LBPM_ycbcr_n8_1D=multiscale_lbp_descriptor_color(img_ycbcr,radius_values=[1,2,3],num_points=8,block_size=8)
            #hist_W_ycbcr_n8=wavelet_descriptor_by_block_size(img_ycbcr,wavelet='haar',block_size=8)
            #DCT for ycbcr
            #hist_DCT_ycbcr_n8_c10_1D=dct_descriptor_color(img_ycbcr, block_size=8, num_coefs=10)
            hist_DCT_ycbcr_n32_c10_1D=dct_descriptor_color(img_ycbcr, block_size=32, num_coefs=10)

            # Extract descriptors from CieLab
            img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
            #LBP for CieLab
            #hist_LBP_LAB_n8_1D=lbp_descriptor_by_block_size(img_LAB,num_points=8,radius=1,block_size=8)
            hist_LBP_LAB_n8_r2_1D=lbp_descriptor_by_block_size(img_LAB,num_points=8,radius=2,block_size=8)
            hist_LBPM_LAB_n8_1D=multiscale_lbp_descriptor_color(img_LAB,radius_values=[1,2,3],num_points=8,block_size=8)
            #hist_W_LAB_n8=wavelet_descriptor_by_block_size(img_LAB,wavelet='haar',block_size=8)
            #DCT for CieLab
            # hist_DCT_LAB_n8_c10_1D=dct_descriptor_color(img_LAB, block_size=8, num_coefs=10)
            hist_DCT_LAB_n32_c10_1D=dct_descriptor_color(img_LAB, block_size=32, num_coefs=10)

            # Extract descriptors for HSV
            img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
            #LBP for HSV
            #hist_LBP_HSV_n8_1D=lbp_descriptor_by_block_size(img_HSV,num_points=8,radius=1,block_size=8)
            hist_LBP_HSV_n8_r2_1D=lbp_descriptor_by_block_size(img_HSV,num_points=8,radius=2,block_size=8)
            hist_LBPM_HSV_n8_1D=multiscale_lbp_descriptor_color(img_HSV,radius_values=[1,2,3],num_points=8,block_size=8)
            #hist_LBPM_HSV_n8_1D=multiscale_lbp_descriptor_color(img_HSV,radius_values=[1,1.5,3],num_points=8,block_size=8)
            #DCT for HSV
            # hist_DCT_HSV_n8_c10_1D=dct_descriptor_color(img_HSV, block_size=8, num_coefs=10)
            hist_DCT_HSV_n32_c10_1D=dct_descriptor_color(img_HSV, block_size=32, num_coefs=10)

            #combine
            # hist_LBP_HSV_LAB_ycbcr_n8_r1_1D = np.concatenate((hist_LBP_HSV_n8_1D, hist_LBP_LAB_n8_1D,hist_LBP_ycbcr_n8_1D))
            hist_LBP_HSV_LAB_ycbcr_n8_r2_1D = np.concatenate((hist_LBP_HSV_n8_r2_1D, hist_LBP_LAB_n8_r2_1D,hist_LBP_ycbcr_n8_r2_1D))
            # hist_DCT_HSV_LAB_ycbcr_n8_c10_1D=np.concatenate((hist_DCT_HSV_n8_c10_1D,hist_DCT_LAB_n8_c10_1D,hist_DCT_ycbcr_n8_c10_1D))
            hist_DCT_HSV_LAB_ycbcr_n32_c10_1D=np.concatenate((hist_DCT_HSV_n32_c10_1D,hist_DCT_LAB_n32_c10_1D,hist_DCT_ycbcr_n32_c10_1D))    

            histograms = {
                'hist_LBP_ycbcr_n8_r2_1D':hist_LBP_ycbcr_n8_r2_1D,
                'hist_LBPM_ycbcr_n8_1D':hist_LBPM_ycbcr_n8_1D,
                'hist_DCT_ycbcr_n32_c10_1D':hist_DCT_ycbcr_n32_c10_1D,
                'hist_LBP_LAB_n8_r2_1D':hist_LBP_LAB_n8_r2_1D,
                'hist_LBPM_LAB_n8_1D':hist_LBPM_LAB_n8_1D,
                'hist_DCT_LAB_n32_c10_1D':hist_DCT_LAB_n32_c10_1D,
                'hist_LBP_HSV_n8_r2_1D':hist_LBP_HSV_n8_r2_1D,
                'hist_LBPM_HSV_n8_1D':hist_LBPM_HSV_n8_1D,
                'hist_DCT_HSV_n32_c10_1D':hist_DCT_HSV_n32_c10_1D,
                'hist_LBP_HSV_LAB_ycbcr_n8_r2_1D':hist_LBP_HSV_LAB_ycbcr_n8_r2_1D,
                'hist_DCT_HSV_LAB_ycbcr_n32_c10_1D':hist_DCT_HSV_LAB_ycbcr_n32_c10_1D
            }
            if flag==1:
                save_path = directory_path + '/week3'
            else:
                save_path=directory_path
            pkl_filename = os.path.splitext(filename)[0] + '_w3.pkl'
            pkl_path = os.path.join(save_path, pkl_filename)
            print(pkl_path)
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(histograms, pkl_file)


# process both folders
directory_query1 = "src/week3/filtered_cropped_qst2_w3"
# directory_query1 = "filtered_cropped_qst2_w3"

directory_query2 = "../../data/BBDD"
print("Current working directory:", os.getcwd())
print("Processing directory 1:")

process_directory(directory_query1, flag=0)

# print("Processing directory 2:")
# process_directory(directory_query2,flag=1)
