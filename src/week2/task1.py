import cv2
import numpy as np
import os
import pickle


def hsv_hist3D(image):
    """
    Compute HSV 3D histogram from an image
    """
    # 180 bins for hue do the the fact that opencv stores hue in 180 bins
    num_bins = [180, 256, 256]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hsv_hist3D = cv2.calcHist([hsv_image], [0, 1, 2], None, num_bins, [0, 180, 0, 256, 0, 256])

    normalized = cv2.normalize(hsv_hist3D, hsv_hist3D, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Flatten the 3D histogram into a 1D vector
    flattened_hist = normalized.flatten()

    return flattened_hist

def lab_hist3D(image):
    """
    Compute Lab 3D histogram from an image and flatten it into a 1D vector.
    """
    # Define number of bins for each channel (L, A, B)
    num_bins = [256, 256, 256]  # 256 bins for each channel (L, A, B)
    
    # Convert image from RGB (OpenCV default BGR) to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    
    # Compute 3D histogram over the Lab channels
    lab_hist3D = cv2.calcHist([lab_image], [0, 1, 2], None, num_bins, [0, 256, 0, 256, 0, 256])
    
    # Normalize the histogram (to the range [0, 1] with NORM_MINMAX)
    normalized = cv2.normalize(lab_hist3D, lab_hist3D, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Flatten the 3D histogram into a 1D vector
    flattened_hist = normalized.flatten()

    return flattened_hist



def spatial_pyramid_histogram(image, levels=2, dimensions=True,hist_size=[8,8], hist_range=[0,256,0,256]):
    """
    Compute a spatial pyramid representation of histograms, with concatenation of histograms per channel.
    Level zero has 1 block. 2^0=1 so blocks 1*1=1
    Level one has 4 blocks. 2^1=2 so blocks 2*2=4
    Level two has 16 blocks. 2^2=4 so blocks 4*4=16
    """

    pyramid_hist = []
    # resize image to 256*256
    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA) # NEEDS DISCUSSION
    h, w = image.shape[:2]  # Get the height and width of the image

    # Loop through each level in the pyramid
    for level in range(levels + 1):
        num_blocks = 2 ** level  
        block_h, block_w = h // num_blocks, w // num_blocks  # Block size

        for i in range(num_blocks):
            for j in range(num_blocks):
                # Define the block region
                block = image[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
                #print(f'block ' + str(i) +' : '+ str(j))
                # Compute histograms depending on the number of channels
                block_hist = []
                if dimensions == True:
                    # Compute 2D histogram 
                    lab_hist2D = cv2.calcHist([block], [0, 1], None, hist_size, hist_range)
                    # Normalize the histogram (to the range [0, 1] with NORM_MINMAX)
                    normalized = cv2.normalize(lab_hist2D, lab_hist2D, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    # Flatten the 2D histogram into a 1D vector
                    flattened_hist = normalized.flatten()
                    block_hist.append(flattened_hist)
                else:
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

            # Extract descriptors from RGB channels
            img_BGR = cv2.imread(img_path)
            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
            hist_RGB_8_2D = spatial_pyramid_histogram(img_RGB, levels=2, dimensions=True,hist_size=[8,8], hist_range=[0, 256,0, 256])
            hist_RGB_8_3D = spatial_pyramid_histogram(img_RGB, levels=2, dimensions=False,hist_size=[8,8,8], hist_range=[0, 256,0, 256,0, 256])
            hist_RGB_128_2D = spatial_pyramid_histogram(img_RGB, levels=2, dimensions=True, hist_size=[128,128], hist_range=[0, 256,0, 256])
            hist_RGB_128_3D = spatial_pyramid_histogram(img_RGB, levels=2, dimensions=False, hist_size=[128,128,128], hist_range=[0, 256,0, 256,0, 256])
            # hist_RGB_256_2D = spatial_pyramid_histogram(img_RGB, levels=2, dimensions=True, hist_size=[256,256], hist_range=[0, 256,0, 256])
            # hist_RGB_256_3D = spatial_pyramid_histogram(img_RGB, levels=2, dimensions=False, hist_size=[256,256,256], hist_range=[0, 256,0, 256,0, 256])

            #CieLab
            img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
            hist_LAB_8_2D = spatial_pyramid_histogram(img_LAB, levels=2, dimensions=True,hist_size=[8,8], hist_range=[0, 256,0, 256])
            hist_LAB_8_3D = spatial_pyramid_histogram(img_LAB, levels=2, dimensions=False,hist_size=[8,8,8], hist_range=[0, 256,0, 256,0, 256])
            hist_LAB_128_2D = spatial_pyramid_histogram(img_LAB, levels=2, dimensions=True, hist_size=[128,128], hist_range=[0, 256,0, 256])
            hist_LAB_128_3D = spatial_pyramid_histogram(img_LAB, levels=2, dimensions=False, hist_size=[128,128,128], hist_range=[0, 256,0, 256,0, 256])
            # hist_LAB_256_2D = spatial_pyramid_histogram(img_LAB, levels=2, dimensions=True, hist_size=[256,256], hist_range=[0, 256,0, 256])
            # hist_LAB_256_3D = spatial_pyramid_histogram(img_LAB, levels=2, dimensions=False, hist_size=[256,256,256], hist_range=[0, 256,0, 256,0, 256])

            #HSV
            img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
            hist_HSV_8_2D = spatial_pyramid_histogram(img_HSV, levels=2, dimensions=True,hist_size=[8,8], hist_range=[0, 180,0, 256])
            hist_HSV_8_3D = spatial_pyramid_histogram(img_HSV, levels=2, dimensions=False,hist_size=[8,8,8], hist_range=[0, 180,0, 256,0, 256])
            hist_HSV_128_2D = spatial_pyramid_histogram(img_HSV, levels=2, dimensions=True, hist_size=[128,128], hist_range=[0, 180,0, 256])
            hist_HSV_128_3D = spatial_pyramid_histogram(img_HSV, levels=2, dimensions=False, hist_size=[128,128,128], hist_range=[0, 180,0, 256,0, 256])
            # hist_HSV_256_2D = spatial_pyramid_histogram(img_HSV, levels=2, dimensions=True, hist_size=[256,256], hist_range=[0, 256,0, 256])
            # hist_HSV_256_3D = spatial_pyramid_histogram(img_HSV, levels=2, dimensions=False, hist_size=[256,256,256], hist_range=[0, 256,0, 256,0, 256])

            histograms = {
                'hist_RGB_8_2D': hist_RGB_8_2D,
                'hist_RGB_8_3D': hist_RGB_8_3D,
                'hist_RGB_128_2D': hist_RGB_128_2D,
                'hist_RGB_128_3D': hist_RGB_128_3D,
                # 'hist_RGB_256_2D': hist_RGB_256_2D,
                # 'hist_RGB_256_3D': hist_RGB_256_3D,
                'hist_LAB_8_2D': hist_LAB_8_2D,
                'hist_LAB_8_3D': hist_LAB_8_3D,
                'hist_LAB_128_2D': hist_LAB_128_2D,
                'hist_LAB_128_3D': hist_LAB_128_3D,
                # 'hist_LAB_256_2D': hist_LAB_256_2D,
                # 'hist_LAB_256_3D': hist_LAB_256_3D,
                'hist_HSV_8_2D': hist_HSV_8_2D,
                'hist_HSV_8_3D': hist_HSV_8_3D,
                'hist_HSV_128_2D': hist_HSV_128_2D,
                'hist_HSV_128_3D': hist_HSV_128_3D,
                # 'hist_HSV_256_2D': hist_HSV_256_2D,
                # 'hist_HSV_256_3D': hist_HSV_256_3D,
            }

            save_path = directory_path + '/week2'
            pkl_filename = os.path.splitext(filename)[0] + '_w2.pkl'
            pkl_path = os.path.join(save_path, pkl_filename)
            print(pkl_path)
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(histograms, pkl_file)



# process both folders
directory_query1 = "datasets/qsd1_w1"
directory_query2 = "data/BBDD"
print("Current working directory:", os.getcwd())
print("Processing directory 1:")
process_directory(directory_query1)

print("Processing directory 2:")
process_directory(directory_query2)

