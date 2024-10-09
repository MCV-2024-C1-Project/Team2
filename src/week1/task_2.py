import os
import pickle
import utils
import pandas as pd

directory = 'data/BBDD/'
df = pd.DataFrame()
list_files_used = []

# catch the fisrt image then the second and so on
for file_compare_image in os.listdir(directory):
    if file_compare_image.endswith('.pkl') and file_compare_image != 'relationships.pkl':
        pkl_grey_path = os.path.join(directory, file_compare_image)
        with open(pkl_grey_path, 'rb') as pkl_file:
            histograms_first = pickle.load(pkl_file)
            list_files_used.append(file_compare_image)

        for filename in os.listdir(directory):
            if filename.endswith('.pkl') and filename != file_compare_image and filename != 'relationships.pkl' and filename not in list_files_used:
                pkl_path = os.path.join(directory, filename)

                with open(pkl_path, 'rb') as pkl_file:
                    histograms = pickle.load(pkl_file)

                histogram_first_grey = histograms_first['grey']
                histogram_grey = histograms['grey']
                index = file_compare_image + '-' + filename
                new_line = {'name': index, 'euc_dist': utils.euc_dist(histogram_first_grey, histogram_grey),
                            'L1_dist': utils.L1_dist(histogram_first_grey, histogram_grey),
                            'X2_distance': utils.X2_distance(histogram_first_grey, histogram_grey),
                            'histogram_similarity': utils.histogram_similiarity(histogram_first_grey, histogram_grey),
                            'hellinger_kernel': utils.hellinger_kernel(histogram_first_grey, histogram_grey)}
                df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)
df.to_csv('grey_distances.csv', index=False)
