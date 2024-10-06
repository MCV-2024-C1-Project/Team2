import re
import os
import pickle
import utils
import numpy as np

directory = 'qsd1_w1'
directory_bbdd = 'data/BBDD/'


def extract_number_from_filename(filename):
    '''Function to extract the number of the image'''
    match = re.search(r'bbdd_(\d+)\.pkl', filename)
    if match:
        return int(match.group(1))


def extract_number_from_filename_qsd1_w1(filename):
    '''Function to extract the number of the image'''
    match = re.search(r'(\d+)\.pkl', filename)
    if match:
        return int(match.group(1))

# Method 1 
min_distance = float('inf')
list_results_k_1 = []
list_results_k_5 = []
list_results_k_10 = []

for file_compare_image in os.listdir(directory):
    # catch the first image then the second and so on of the qsd1_w1

    if file_compare_image.endswith('.pkl') and file_compare_image != 'gt_corresps.pkl':
        pkl_grey_path = os.path.join(directory, file_compare_image)
        with open(pkl_grey_path, 'rb') as pkl_file:
            histograms_first = pickle.load(pkl_file)

        distances = []
        index_qsd1_w1 = extract_number_from_filename_qsd1_w1(file_compare_image)

        for filename in os.listdir(directory_bbdd):
            if filename.endswith('.pkl') and filename != 'relationships.pkl':
                pkl_path = os.path.join(directory_bbdd, filename)

                with open(pkl_path, 'rb') as pkl_file:
                    histograms = pickle.load(pkl_file)
                # distancess = []
                # for i in range(3):  
                #     distance = utils.euc_dist(histograms_first['hist_LAB'][i], histograms['hist_LAB'][i])
                #     distancess.append(distance)
                histogram_first_grey = histograms_first['hist_LAB']
                histogram_grey = histograms['hist_LAB']
                # # Try all 4 loss functions: euc_dist, L1_dist, X2_distance, hellinger_kernel, histogram_similiarity
                distance = utils.our_metric(histogram_first_grey, histogram_grey)
                #distancia = np.mean(distancess)

                index = extract_number_from_filename(filename)

                distances.append((distance, index))

                # Sort the distances and select the top k results
        distances.sort(key=lambda x: x[0])  # Sort by distance (lowest first)
        top_k_1_result = [distances[0][1]] 
        top_k_5_results = [index for _, index in distances[:5]]  
        top_k_10_results = [index for _, index in distances[:10]] 

        # when the loop for the first query image is finish we save the index in a list
        list_results_k_1.append(top_k_1_result)
        list_results_k_5.append(top_k_5_results)
        list_results_k_10.append(top_k_10_results)


list_results_k_1 = [result for result in list_results_k_1]  
list_results_k_5 = [result for result in list_results_k_5]  
list_results_k_10 = [result for result in list_results_k_10]  

# Para los resultados aplanados
predicted_flattened_k_5 = [p for p in list_results_k_5]
predicted_flattened_k_10 = [p for p in list_results_k_10]


with open(f'qsd1_w1/gt_corresps.pkl', 'rb') as f:
    ground_truth = pickle.load(f)

print("Method 1 - LAB with our metric; k=1 mapk:", utils.mapk(ground_truth, list_results_k_1, k=1))
print("Method 1 - LAB with our metric; k=5 mapk:", utils.mapk(ground_truth, predicted_flattened_k_5, k=5))

#To ensure that they are integers
list_results_k_10_enteros = [[int(x) for x in sublist] for sublist in list_results_k_10]

# store results k=10
with open('results/week1/method1/result.pkl', 'wb') as pkl_file:
    pickle.dump(list_results_k_10_enteros, pkl_file)

# Method 2
min_distance = float('inf')
list_results_k_1 = []
list_results_k_5 = []
list_results_k_10 = []

for file_compare_image in os.listdir(directory):
    # catch the first image then the second and so on of the qsd1_w1

    if file_compare_image.endswith('.pkl') and file_compare_image != 'gt_corresps.pkl':
        pkl_grey_path = os.path.join(directory, file_compare_image)
        with open(pkl_grey_path, 'rb') as pkl_file:
            histograms_first = pickle.load(pkl_file)

        distances = []
        index_qsd1_w1 = extract_number_from_filename_qsd1_w1(file_compare_image)

        for filename in os.listdir(directory_bbdd):
            if filename.endswith('.pkl') and filename != 'relationships.pkl':
                pkl_path = os.path.join(directory_bbdd, filename)

                with open(pkl_path, 'rb') as pkl_file:
                    histograms = pickle.load(pkl_file)
               
                histogram_first_grey = histograms_first['hist_HSV']
                histogram_grey = histograms['hist_HSV']
                # # Try all 4 loss functions: euc_dist, L1_dist, X2_distance, hellinger_kernel, histogram_similiarity
                distance = utils.our_metric(histogram_first_grey, histogram_grey)

                index = extract_number_from_filename(filename)

                distances.append((distance, index))

        # Sort the distances and select the top k results
        distances.sort(key=lambda x: x[0])  # Sort by distance (lowest first)
        top_k_1_result = [distances[0][1]] 
        top_k_5_results = [index for _, index in distances[:5]]  
        top_k_10_results = [index for _, index in distances[:10]] 

        # when the loop for the first query image is finish we save the index in a list
        list_results_k_1.append(top_k_1_result)
        list_results_k_5.append(top_k_5_results)
        list_results_k_10.append(top_k_10_results)


list_results_k_1 = [result for result in list_results_k_1]  
list_results_k_5 = [result for result in list_results_k_5]  
list_results_k_10 = [result for result in list_results_k_10]  

# Para los resultados aplanados
predicted_flattened_k_5 = [p for p in list_results_k_5]
predicted_flattened_k_10 = [p for p in list_results_k_10]

with open(f'qsd1_w1/gt_corresps.pkl', 'rb') as f:
    ground_truth = pickle.load(f)

print("Method 2 - HSV with our metric; k=1 mapk:", utils.mapk(ground_truth, list_results_k_1, k=1))
print("Method 2 - HSV with our metric; k=5 mapk:", utils.mapk(ground_truth, predicted_flattened_k_5, k=5))

#To ensure that they are integers
list_results_k_10_enteros = [[int(x) for x in sublist] for sublist in list_results_k_10]

# store results k=10
with open('results/week1/method2/result.pkl', 'wb') as pkl_file:
    pickle.dump(list_results_k_10_enteros, pkl_file)


# Method HSV with X2_distance
min_distance = float('inf')
list_results_k_1 = []
list_results_k_5 = []
list_results_k_10 = []

for file_compare_image in os.listdir(directory):
    # catch the first image then the second and so on of the qsd1_w1

    if file_compare_image.endswith('.pkl') and file_compare_image != 'gt_corresps.pkl':
        pkl_grey_path = os.path.join(directory, file_compare_image)
        with open(pkl_grey_path, 'rb') as pkl_file:
            histograms_first = pickle.load(pkl_file)

        distances = []
        index_qsd1_w1 = extract_number_from_filename_qsd1_w1(file_compare_image)

        for filename in os.listdir(directory_bbdd):
            if filename.endswith('.pkl') and filename != 'relationships.pkl':
                pkl_path = os.path.join(directory_bbdd, filename)

                with open(pkl_path, 'rb') as pkl_file:
                    histograms = pickle.load(pkl_file)
                # distancess = []
                # for i in range(3):  
                #     distance = utils.euc_dist(histograms_first['hist_LAB'][i], histograms['hist_LAB'][i])
                #     distancess.append(distance)
                histogram_first_grey = histograms_first['hist_HSV']
                histogram_grey = histograms['hist_HSV']
                # # Try all 4 loss functions: euc_dist, L1_dist, X2_distance, hellinger_kernel, histogram_similiarity
                distance = utils.X2_distance(histogram_first_grey, histogram_grey)
                #distancia = np.mean(distancess)

                index = extract_number_from_filename(filename)

                distances.append((distance, index))

                # Sort the distances and select the top k results
        distances.sort(key=lambda x: x[0])  # Sort by distance (lowest first)
        top_k_1_result = [distances[0][1]] 
        top_k_5_results = [index for _, index in distances[:5]]  
        top_k_10_results = [index for _, index in distances[:10]] 

        # when the loop for the first query image is finish we save the index in a list
        list_results_k_1.append(top_k_1_result)
        list_results_k_5.append(top_k_5_results)
        list_results_k_10.append(top_k_10_results)


list_results_k_1 = [result for result in list_results_k_1]  
list_results_k_5 = [result for result in list_results_k_5]  
list_results_k_10 = [result for result in list_results_k_10]  

# Para los resultados aplanados
predicted_flattened_k_5 = [p for p in list_results_k_5]
predicted_flattened_k_10 = [p for p in list_results_k_10]


with open(f'qsd1_w1/gt_corresps.pkl', 'rb') as f:
    ground_truth = pickle.load(f)

print("HSV with X2_distance; k=1 mapk:", utils.mapk(ground_truth, list_results_k_1, k=1))
print("HSV with X2_distance; k=5 mapk:", utils.mapk(ground_truth, predicted_flattened_k_5, k=5))


# Method LAB with X2_distance
min_distance = float('inf')
list_results_k_1 = []
list_results_k_5 = []
list_results_k_10 = []

for file_compare_image in os.listdir(directory):
    # catch the first image then the second and so on of the qsd1_w1

    if file_compare_image.endswith('.pkl') and file_compare_image != 'gt_corresps.pkl':
        pkl_grey_path = os.path.join(directory, file_compare_image)
        with open(pkl_grey_path, 'rb') as pkl_file:
            histograms_first = pickle.load(pkl_file)

        distances = []
        index_qsd1_w1 = extract_number_from_filename_qsd1_w1(file_compare_image)

        for filename in os.listdir(directory_bbdd):
            if filename.endswith('.pkl') and filename != 'relationships.pkl':
                pkl_path = os.path.join(directory_bbdd, filename)

                with open(pkl_path, 'rb') as pkl_file:
                    histograms = pickle.load(pkl_file)
               
                histogram_first_grey = histograms_first['hist_LAB']
                histogram_grey = histograms['hist_LAB']
                # # Try all 4 loss functions: euc_dist, L1_dist, X2_distance, hellinger_kernel, histogram_similiarity
                distance = utils.X2_distance(histogram_first_grey, histogram_grey)

                index = extract_number_from_filename(filename)

                distances.append((distance, index))

        # Sort the distances and select the top k results
        distances.sort(key=lambda x: x[0])  # Sort by distance (lowest first)
        top_k_1_result = [distances[0][1]] 
        top_k_5_results = [index for _, index in distances[:5]]  
        top_k_10_results = [index for _, index in distances[:10]] 

        # when the loop for the first query image is finish we save the index in a list
        list_results_k_1.append(top_k_1_result)
        list_results_k_5.append(top_k_5_results)
        list_results_k_10.append(top_k_10_results)


list_results_k_1 = [result for result in list_results_k_1]  
list_results_k_5 = [result for result in list_results_k_5]  
list_results_k_10 = [result for result in list_results_k_10]  

# Para los resultados aplanados
predicted_flattened_k_5 = [p for p in list_results_k_5]
predicted_flattened_k_10 = [p for p in list_results_k_10]

with open(f'qsd1_w1/gt_corresps.pkl', 'rb') as f:
    ground_truth = pickle.load(f)

print("LAB with X2_distance; k=1 mapk:", utils.mapk(ground_truth, list_results_k_1, k=1))
print("LAB with X2_distance; k=5 mapk:", utils.mapk(ground_truth, predicted_flattened_k_5, k=5))

