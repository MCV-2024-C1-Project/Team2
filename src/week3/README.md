# Team2 

## MCV, C1: Introduction to Human and Computer Vision - week3

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

