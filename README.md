# Team2

## MCV, C1: Introduction to Human and Computer Vision - week1

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
