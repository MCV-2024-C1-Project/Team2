import cv2
import numpy as np
import os
import pickle

def spatial_pyramid_histogram(image, levels=2, hist_size=8, hist_range=[0,256]):
    """
    Compute a spatial pyramid representation of histograms, with concatenation of histograms per channel.
    Level zero has 1 block. 2^0=1 so blocks 1*1=1
    Level one has 4 blocks. 2^1=2 so blocks 2*2=4
    Level two has 16 blocks. 2^2=4 so blocks 4*4=16
    """

    pyramid_hist = []
    h, w = image.shape[:2]  # Get the height and width of the image
    channels = 1 if len(image.shape) == 2 else image.shape[2]  # Check number of channels

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
                if channels == 1:
                    # Single-channel image (grayscale)
                    hist = cv2.calcHist([block], [0], None, [hist_size], hist_range)
                    hist /= hist.sum()  # Normalize the histogram
                    block_hist.append(hist.flatten())
                else:
                    # Multi-channel image (e.g., BGR)
                    for ch in range(channels):
                        #print(f'channel'+str(ch))
                        hist = cv2.calcHist([block], [ch], None, [hist_size], hist_range)
                        hist /= hist.sum()  # Normalize the histogram
                        block_hist.append(hist.flatten())

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
            # Extract descriptors from image greyscale
            img_grey = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Compute spatial pyramid histogram 
            hist_grey_8 = spatial_pyramid_histogram(img_grey, levels=2, hist_size=8, hist_range=[0, 256])
            hist_grey_128 = spatial_pyramid_histogram(img_grey, levels=2, hist_size=128, hist_range=[0, 256])
            hist_grey_256 = spatial_pyramid_histogram(img_grey, levels=2, hist_size=256, hist_range=[0, 256])

            # Extract descriptors from RGB channels
            img_BGR = cv2.imread(img_path)
            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
            hist_RGB_8 = spatial_pyramid_histogram(img_RGB, levels=2, hist_size=8, hist_range=[0, 256])
            hist_RGB_128 = spatial_pyramid_histogram(img_RGB, levels=2, hist_size=128, hist_range=[0, 256])
            hist_RGB_256 = spatial_pyramid_histogram(img_RGB, levels=2, hist_size=256, hist_range=[0, 256])

            #CieLab
            img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
            hist_LAB_8 = spatial_pyramid_histogram(img_LAB, levels=2, hist_size=8, hist_range=[0, 256])
            hist_LAB_128 = spatial_pyramid_histogram(img_LAB, levels=2, hist_size=128, hist_range=[0, 256])
            hist_LAB_256 = spatial_pyramid_histogram(img_LAB, levels=2, hist_size=256, hist_range=[0, 256])

            #HSV
            img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
            hist_HSV_8 = spatial_pyramid_histogram(img_HSV, levels=2, hist_size=8, hist_range=[0, 256])
            hist_HSV_128 = spatial_pyramid_histogram(img_HSV, levels=2, hist_size=128, hist_range=[0, 256])
            hist_HSV_256 = spatial_pyramid_histogram(img_HSV, levels=2, hist_size=256, hist_range=[0, 256])


            histograms = {
                'hist_grey_8': hist_grey_8,
                'hist_grey_128': hist_grey_128,
                'hist_grey_256': hist_grey_256,
                'hist_RGB_8': hist_RGB_8,
                'hist_RGB_128': hist_RGB_128,
                'hist_RGB_256': hist_RGB_256,
                'hist_LAB_8': hist_LAB_8,
                'hist_LAB_128': hist_LAB_128,
                'hist_LAB_256': hist_LAB_256,
                'hist_HSV_8': hist_HSV_8,
                'hist_HSV_128': hist_HSV_128,
                'hist_HSV_256': hist_HSV_256,
            }

            # Verify if the folder is BBDD to add '_w2' to the pickle path so to no delete the pickles from the last week
            if directory_path == "../../data/BBDD":
                pkl_filename = os.path.splitext(filename)[0] + '_w2.pkl'
            else:
                pkl_filename = os.path.splitext(filename)[0] + '.pkl'

            pkl_path = os.path.join(directory_path, pkl_filename)
            #print(pkl_path)
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(histograms, pkl_file)

# process both folders
directory_query1 = "../../datasets/qsd1_w2/"
directory_query2 = "../../data/BBDD"

print("Processing directory 1:")
process_directory(directory_query1)

print("Processing directory 2:")
process_directory(directory_query2)

