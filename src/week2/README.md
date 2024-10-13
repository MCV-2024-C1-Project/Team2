# Team2 

## MCV, C1: Introduction to Human and Computer Vision - week2

## Project Description

This project focuses on image retrieval and background removal tasks using color histograms and mask-based segmentation techniques. The goal is to evaluate and enhance image retrieval systems by comparing different feature extraction methods, such as block-based and hierarchical histograms, and applying them to query datasets.

The project is divided into several key tasks:

1. **Feature Extraction**: Implementing various histogram-based methods (2D, 3D, block-based, and hierarchical) to represent image features effectively for retrieval.
  
2. **Background Removal**: Using color thresholds to remove the background in images, creating binary masks that help isolate the foreground, which improves retrieval accuracy.
  
3. **Performance Evaluation**: Testing the developed image retrieval system on two different query sets (cropped and background-containing images), and evaluating the system's performance using retrieval metrics like precision, recall, and F1-measure.


---

## Dataset Information

### Museum Datasets:
1. **Can Framis Museum**
2. **Figueres 120 Years Expo**
3. **Kode Bergen**

---

## Tasks

### Task 1: Histogram-Based Descriptor Implementations
- **3D/2D Histograms**: Implement both 3D and 2D histograms for color feature extraction.
- **Block-Based Histograms**: 
    - Divide each image into non-overlapping blocks.
    - Compute histograms for each block and concatenate them to form the descriptor.
- **Hierarchical Histograms (Spatial Pyramid Representation)**: 
    - Compute block histograms at different pyramid levels (e.g., Level 1, Level 2, Level 3).
    - Concatenate the histograms across levels for enhanced representation.

### Task 2: Query System Testing with QSD1-W2
- Test the query system using **QSD1-W2** development set and evaluate the retrieval results.
- Use the best-performing descriptor from **Task 1**.
- Compare results against the best descriptor used in Week 1.

### Task 3: Background Removal for QSD2-W2
- Remove the background from each image in **QSD2-W2** using a color threshold or background distribution model.
- Steps:
    1. Create a binary mask to segment the foreground from the background.
    2. Compute descriptors using only the foreground pixels.
    3. Avoid using contour or object detectors; only rely on color thresholds.

### Task 4: Evaluation of Background Removal
- Evaluate the background removal process using **Precision**, **Recall**, and **F1-measure**.
- Metrics:
    - **True Positives (TP)**: Foreground pixels correctly detected.
    - **False Positives (FP)**: Background pixels incorrectly classified as foreground.
    - **False Negatives (FN)**: Foreground pixels classified as background.
    - **True Negatives (TN)**: Background pixels correctly classified.

### Task 5: Background Removal and Retrieval for QSD2-W2
- After background removal, apply the image retrieval system for **QSD2-W2** and return correspondences for each painting.
- Only the retrieval results will be evaluated for this task.

### Task 6: Submission for Blind Competition
- **QST1-W2** and **QST2-W2**: For each test query, submit a list of the **K=10** best results.
- Submission Format:
    - Create a Python list of lists containing image IDs (as integers).
    - For **QST2-W2**, return binary masks with the same name as the query image but in `.png` format.
        - Example: `00001.jpg` → `00001.png`

--- 

### Requirements
- Python 3.x
- NumPy, Matplotlib, OpenCV, pickle
- Additional libraries: [benhamner/Metrics](https://github.com/benhamner/Metrics) for computing mAP@K

--- 

### File Structure
- project_root/
  - data/
    - bbdd/               # Museum dataset
  - datasets/
    - qsd2_w2/              # Query dataset

  - src/
    - week1/
      - `create_database.py`       # Code for loading images, extracting descriptors, and saving as a .pkl file.
      - `task_2.py`        # Code for similarity measures
      - `task_3.py, task_4.py`         # Code for retrieving top K images
    - week2/
      - `task1.py`     # Computes and saves spatial pyramid histograms for images in RGB, LAB, and HSV.
      - `task2.py`      # Computes MAP@k results for histogram comparisons and saves them to a CSV.
      - `task6_qst1_w2.py`   # Implements background removal techniques for images and retrieves the top 10 similar images based on computed histograms and distance metrics.
  - results/
    - week1/
        - QST1/method1/result.pkl     # Retrieval results for QSD1 method1
        - QST1/method2/result.pkl     # Retrieval results for QST1 method2
    - week2/
        - QST1/method1/result.pkl # Retrieval results for QSD1 method1
        - QST2/method1/result.pkl # Retrieval results for QSD2 method1
  
