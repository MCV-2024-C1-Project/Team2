import pickle

with open("../../results/week3/QST2/method1/result.pkl", 'rb') as pkl_file:
    result = pickle.load(pkl_file)
    print(result)
