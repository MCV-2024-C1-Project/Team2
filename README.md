# Team2

## MCV, C1: Introduction to Human and Computer Vision 

### Week 1

## Project Description
This project focuses on **Content-Based Image Retrieval (CBIR)**, where the main goal is to build a retrieval system that allows users to search for paintings from a museum image collection using a query-by-example approach. The system uses image descriptors, such as histograms, to extract features based on color, texture, and other visual characteristics, and then retrieves the most similar images from the dataset.

## Tasks Overview

### Task 1: Compute Image Descriptors
- **Image Descriptor Methods**: 
  - Use **1D color histograms** (gray-level histograms or color component concatenation) in various color spaces (RGB, CieLab, YCbCr, HSV).
  - Compute descriptors for both the **museum dataset (BBDD)** and the **query set (QSD1)**.

### Task 2: Similarity Measures
- Implement similarity measures for comparing image descriptors:
  - **L1 Distance**
  - **Chi-Square (χ²) Distance**

### Task 3: Retrieval System
- For each query image in **QSD1**, compute the similarities between the query image and museum images (BBDD).
- Retrieve the **K most similar images** and rank them by score (lowest distance).
- Evaluate performance using **mAP@K**, with specific focus on **K=1** and **K=5**.

## Evaluation Metrics
- **mAP@K (mean Average Precision)** will be used to measure retrieval accuracy.
  - **AP@K (Average Precision)** is computed for each query.
  - **mAP@K**: Mean over all queries.
- Performance will be measured at **K=1** and **K=5**.

## Dataset Information
- **Museum Dataset (BBDD)**: A collection of paintings provided by the course.
- **Query Set (QSD1)**: Cropped images with minimal rotation. Ground truth is provided.
- **Test Set (QST1)**: Similar to QSD1 but without ground truth, used for the blind challenge.

## How to Run the Project

### Requirements
- Python 3.x
- NumPy, Matplotlib, OpenCV, pickle
- Additional libraries: [benhamner/Metrics](https://github.com/benhamner/Metrics) for computing mAP@K

### Instructions

1. **Run Main Script**:
   - Start by running the `main.py` script to initialize and prepare the environment for the following steps.
     
2. **Compute Descriptors**:
   - Run the `create_database.py` script to generate descriptors for both **museum** and **query images**.
   
3. **Similarity Measures**:
   - Use the implemented similarity functions to compute distances between query and museum descriptors (`task2.py`).

4. **Retrieve Top K Images**:
   - For each query image, retrieve and rank the **top K most similar museum images**. (`task_3&4.py`)



### File Structure
- project_root/
  - data/
    - bbdd/               # Museum dataset
  - qsd1_w1/              # Query dataset

  - src/
    - `create_database.py`       # Code for loading images, extracting descriptors, and saving as a .pkl file.
    - `task2.py`        # Code for similarity measures
    - `task_3&4.py`         # Code for retrieving top K images
  - results/
    - week1/
        - method1/result.pkl     # Retrieval results for QSD1 method1
        - method2/result.pkl     # Retrieval results for QST1 method2


### Week 2

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
  


## Week 3

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#Requirements)
- [Tasks Overview](#tasks-overview)
- [Evaluation Metrics](#evaluation-metrics)


## Introduction
This repository contains implementations of various image processing and retrieval tasks, including noise filtering, texture-based feature extraction, and retrieval methods. The project focuses on analyzing and processing a set of images using descriptors and noise filtering techniques.

## Requirements
- Python 3.x
- NumPy, Matplotlib, OpenCV, pickle
- Additional libraries: [benhamner/Metrics](https://github.com/benhamner/Metrics) for computing mAP@K

## Tasks Overview

### Task 1: Noise Filtering
#### Objective: Filter out noise from images using linear and non-linear filters.
- **Noise model**: Unknown (only some images contain noise).
- **Approach**: Apply different noise filters depending on whether the image is noisy.
- **Example Images**: Uses examples from the QSD1-W3 dataset.


### Task 2: Texture Descriptors-Based Retrieval System
#### Task 2(I): Texture Descriptor Implementation
#### Objective: Implement various texture descriptors such as Local Binary Patterns (LBP), Discrete Cosine Transform (DCT), and wavelet-based descriptors.
- **Evaluation**: Assess retrieval performance on QSD1-W3 dataset using only texture descriptors.

#### Task 2(II): DCT Descriptor Implementation
#### Descriptor Flow:
  1) Apply 2D DCT on each block of the image.
  2) Perform a zig-zag scan to order DCT coefficients.
  3) Keep the first `N` coefficients to form a block feature vector.
  4) Concatenate block vectors into a single feature vector for each image.

#### Task 2(III): LBP Descriptor Implementation
#### Descriptor Flow:
  1) Divide image into blocks.
  2) Compute binary patterns for each pixel based on neighborhood values.
  3) Compute histograms of patterns in each block.
  4) Multiscale LBP: Uses multiple local neighborhoods with bilinear interpolation for off-pixel points.

### Task 3: Painting Detection and Background Removal
#### Objective: Detect all paintings (up to two per image) in QSD2-W3 and remove the background.
- **Method**: Generates a binary mask for evaluation and comparison.

### Task 4: Background Removal with Retrieval and Correspondence
#### Objective: Detect paintings, remove backgrounds, and apply a retrieval system on QSD2-W3.
- **Output**: Return correspondences between query and database paintings.

### Task 5: Competition Submission (QST1 & QST2)
- **Objective**: Submit segmentation masks for a "blind" competition.
- **Structure**: Use an extra list level in data structure for supporting multiple paintings per image.


## Evaluation Metrics
#### The following metrics are calculated to evaluate the performance of the mask generation techniques:
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to the all actual positives.
- **F1 Score**: The weighted average of Precision and Recall.

