import numpy as np
import os
import pickle
import cv2
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct


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


# def bilinear_interpolation(img, x, y):
#     h, w = img.shape
    
#     # Calcular los índices de los píxeles vecinos
#     x1, y1 = int(x), int(y)
    
#     # Verificar los límites de los índices
#     x2 = x1 + 1 if x1 + 1 < w else x1  # Si está en el borde, usar el mismo índice
#     y2 = y1 + 1 if y1 + 1 < h else y1  # Si está en el borde, usar el mismo índice

#     # Distancias fraccionales
#     dx, dy = x - x1, y - y1

#     # Interpolación bilineal
#     R1 = (1 - dx) * img[y1, x1] + dx * img[y1, x2]
#     R2 = (1 - dx) * img[y2, x1] + dx * img[y2, x2]
#     P = (1 - dy) * R1 + dy * R2

#     return P

# def multiscale_lbp_descriptor_color_2(image, radius_range=[1, 2, 3], num_points=8, block_size=8):
#     fixed_size = (256, 256)
#     resized_image = cv2.resize(image, fixed_size)
#     h, w = resized_image.shape[:2]
#     lbp_blocks = []

#     for channel in range(3):
#         channel_image = image[:, :, channel]  # Extraer canal
#         channel_blocks = []
        
#         # Dividir la imagen en bloques
#         for i in range(0, h, block_size):
#             for j in range(0, w, block_size):
#                 block = channel_image[i:i + block_size, j:j + block_size]
#                 if block.shape == (block_size, block_size):
#                     block_hist = []
#                     # Calcular LBP en múltiples escalas
#                     for radius in radius_range:
#                         lbp = np.zeros_like(block)
#                         for y in range(block.shape[0]):
#                             for x in range(block.shape[1]):
#                                 center = block[y, x]
#                                 for p in range(num_points):
#                                     theta = 2 * np.pi * p / num_points
#                                     neighbor_x = x + radius * np.cos(theta)
#                                     neighbor_y = y - radius * np.sin(theta)

#                                     # Interpolación bilineal para obtener el valor del vecino
#                                     neighbor_value = bilinear_interpolation(block, neighbor_x, neighbor_y)

#                                     # Comparar vecino con el valor del píxel central
#                                     lbp[y, x] += (neighbor_value > center) << p

#                         lbp_uint8 = np.uint8(lbp)
#                         hist, _ = np.histogram(lbp_uint8.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
#                         hist = hist / np.sum(hist)  # Normalizar histograma
#                         block_hist.append(hist)
                    
#                     # Concatenar los histogramas de distintas escalas
#                     channel_blocks.append(np.concatenate(block_hist))
        
#         # Concatenar los histogramas de todos los bloques para este canal
#         lbp_blocks.append(np.concatenate(channel_blocks))

#     # Concatenar los descriptores de los 3 canales
#     feature_vector = np.concatenate(lbp_blocks)
#     return feature_vector

# def multiscale_lbp_descriptor_color(image, radius_values, num_points=8, block_size=8):
#     h, w, _ = image.shape
#     lbp_descriptors = []

#     # Separar los canales de la imagen
#     channels = cv2.split(image)

#     # Para cada canal de color (R, G, B)
#     for channel in channels:
#         channel_descriptors = []
        
#         # Dividir la imagen en bloques
#         for i in range(0, h, block_size):
#             for j in range(0, w, block_size):
#                 block = channel[i:i + block_size, j:j + block_size]
                
#                 # Si el bloque no es de tamaño completo, lo rellenamos con ceros
#                 if block.shape != (block_size, block_size):
#                     padded_block = np.zeros((block_size, block_size), dtype=block.dtype)
#                     padded_block[:block.shape[0], :block.shape[1]] = block
#                     block = padded_block
                
#                 # Calcular LBP en múltiples escalas
#                 for radius in radius_values:
#                     # Redimensionar bloque con interpolación bilineal
#                     scale_factor = radius / min(radius_values)
#                     scaled_block = cv2.resize(block, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                    
#                     # Aplicar LBP
#                     lbp = local_binary_pattern(scaled_block, num_points, radius, method="uniform")
#                     lbp_uint8 = np.uint8(lbp)
                    
#                     # Obtener histograma del LBP
#                     hist, _ = np.histogram(lbp_uint8.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
#                     hist = hist / np.sum(hist)  # Normalizar el histograma
                    
#                     # Agregar histograma al descriptor del bloque
#                     channel_descriptors.append(hist)
        
#         # Concatenar los descriptores LBP para este canal
#         lbp_descriptors.append(np.concatenate(channel_descriptors))
    
#     # Concatenar los descriptores de los tres canales
#     feature_vector = np.concatenate(lbp_descriptors)
#     return feature_vector


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
            hist_LBP_ycbcr_n8_1D=lbp_descriptor_by_block_size(img_ycbcr,num_points=8,radius=1,block_size=8)
            hist_LBP_ycbcr_n8_r2_1D=lbp_descriptor_by_block_size(img_ycbcr,num_points=8,radius=2,block_size=8)
            #hist_LBPM_ycbcr_n8_1D_2=multiscale_lbp_descriptor_color_2(img_ycbcr,radius_range=[1,2.5,4],num_points=8,block_size=8)
            #hist_LBPM_ycbcr_n8_1D=multiscale_lbp_descriptor_color(img_ycbcr,radius_values=[1,1.5,3],num_points=8,block_size=8)
            #DCT for ycbcr
            hist_DCT_ycbcr_n8_c10_1D=dct_descriptor_color(img_ycbcr, block_size=8, num_coefs=10)
            hist_DCT_ycbcr_n32_c10_1D=dct_descriptor_color(img_ycbcr, block_size=32, num_coefs=10)

            # Extract descriptors from CieLab
            img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
            #LBP for CieLab
            hist_LBP_LAB_n8_1D=lbp_descriptor_by_block_size(img_LAB,num_points=8,radius=1,block_size=8)
            hist_LBP_LAB_n8_r2_1D=lbp_descriptor_by_block_size(img_LAB,num_points=8,radius=2,block_size=8)
            #hist_LBPM_LAB_n8_1D_2=multiscale_lbp_descriptor_color_2(img_LAB,radius_range=[1,2.5,4],num_points=8,block_size=8)
            #hist_LBPM_LAB_n8_1D=multiscale_lbp_descriptor_color(img_LAB,radius_values=[1,1.5,3],num_points=8,block_size=8)
            #DCT for CieLab
            hist_DCT_LAB_n8_c10_1D=dct_descriptor_color(img_LAB, block_size=8, num_coefs=10)
            hist_DCT_LAB_n32_c10_1D=dct_descriptor_color(img_LAB, block_size=32, num_coefs=10)

            # Extract descriptors for HSV
            img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
            #LBP for HSV
            hist_LBP_HSV_n8_1D=lbp_descriptor_by_block_size(img_HSV,num_points=8,radius=1,block_size=8)
            hist_LBP_HSV_n8_r2_1D=lbp_descriptor_by_block_size(img_HSV,num_points=8,radius=2,block_size=8)
            #hist_LBPM_HSV_n8_1D_2=multiscale_lbp_descriptor_color_2(img_HSV,radius_range=[1,2.5,4],num_points=8,block_size=8)
            #hist_LBPM_HSV_n8_1D=multiscale_lbp_descriptor_color(img_HSV,radius_values=[1,1.5,3],num_points=8,block_size=8)
            #DCT for HSV
            hist_DCT_HSV_n8_c10_1D=dct_descriptor_color(img_HSV, block_size=8, num_coefs=10)
            hist_DCT_HSV_n32_c10_1D=dct_descriptor_color(img_HSV, block_size=32, num_coefs=10)

            #combine
            hist_LBP_HSV_LAB_ycbcr_n8_r1_1D = np.concatenate((hist_LBP_HSV_n8_1D, hist_LBP_LAB_n8_1D,hist_LBP_ycbcr_n8_1D))
            hist_LBP_HSV_LAB_ycbcr_n8_r2_1D = np.concatenate((hist_LBP_HSV_n8_r2_1D, hist_LBP_LAB_n8_r2_1D,hist_LBP_ycbcr_n8_r2_1D))
            hist_DCT_HSV_LAB_ycbcr_n8_c10_1D=np.concatenate((hist_DCT_HSV_n8_c10_1D,hist_DCT_LAB_n8_c10_1D,hist_DCT_ycbcr_n8_c10_1D))
            hist_DCT_HSV_LAB_ycbcr_n32_c10_1D=np.concatenate((hist_DCT_HSV_n32_c10_1D,hist_DCT_LAB_n32_c10_1D,hist_DCT_ycbcr_n32_c10_1D))    

            histograms = {
                'hist_LBP_ycbcr_n8_1D':hist_LBP_ycbcr_n8_1D,
                'hist_LBP_ycbcr_n8_r2_1D':hist_LBP_ycbcr_n8_r2_1D,
                'hist_DCT_ycbcr_n8_c10_1D':hist_DCT_ycbcr_n8_c10_1D,
                'hist_DCT_ycbcr_n32_c10_1D':hist_DCT_ycbcr_n32_c10_1D,
                'hist_LBP_LAB_n8_1D':hist_LBP_LAB_n8_1D,
                'hist_LBP_LAB_n8_r2_1D':hist_LBP_LAB_n8_r2_1D,
                'hist_DCT_LAB_n8_c10_1D':hist_DCT_LAB_n8_c10_1D,
                'hist_DCT_LAB_n32_c10_1D':hist_DCT_LAB_n32_c10_1D,
                'hist_LBP_HSV_n8_1D':hist_LBP_HSV_n8_1D,
                'hist_LBP_HSV_n8_r2_1D':hist_LBP_HSV_n8_r2_1D,
                'hist_DCT_HSV_n8_c10_1D':hist_DCT_HSV_n8_c10_1D,
                'hist_DCT_HSV_n32_c10_1D':hist_DCT_HSV_n32_c10_1D,
                'hist_LBP_HSV_LAB_ycbcr_n8_r1_1D':hist_LBP_HSV_LAB_ycbcr_n8_r1_1D,
                'hist_LBP_HSV_LAB_ycbcr_n8_r2_1D':hist_LBP_HSV_LAB_ycbcr_n8_r2_1D,
                'hist_DCT_HSV_LAB_ycbcr_n8_c10_1D':hist_DCT_HSV_LAB_ycbcr_n8_c10_1D,
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
directory_query1 = "filtered_images"
directory_query2 = "../../data/BBDD"
print("Current working directory:", os.getcwd())
print("Processing directory 1:")

process_directory(directory_query1,flag=0)

print("Processing directory 2:")
process_directory(directory_query2,flag=1)

