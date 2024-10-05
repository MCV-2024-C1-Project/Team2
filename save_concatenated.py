import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_histograms(img_path):
    # Read the image
    img_BGR = cv2.imread(img_path)

    # HSV Histograms
    img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([img_HSV], [0], None, [256], [0, 256])
    hist_h /= hist_h.sum()
    hist_s = cv2.calcHist([img_HSV], [1], None, [256], [0, 256])
    hist_s /= hist_s.sum()
    hist_v = cv2.calcHist([img_HSV], [2], None, [256], [0, 256])
    hist_v /= hist_v.sum()

    # LAB Histograms
    img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
    hist_l = cv2.calcHist([img_LAB], [0], None, [256], [0, 256])
    hist_l /= hist_l.sum()
    hist_a = cv2.calcHist([img_LAB], [1], None, [256], [0, 256])
    hist_a /= hist_a.sum()
    hist_b_lab = cv2.calcHist([img_LAB], [2], None, [256], [0, 256])
    hist_b_lab /= hist_b_lab.sum()

    # Concatenate HSV and LAB histograms
    hist_hsv = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    hist_lab = np.concatenate([hist_l.flatten(), hist_a.flatten(), hist_b_lab.flatten()])

    # Concatenated histogram of HSV and LAB
    hist_concat = np.concatenate([hist_hsv, hist_lab])

    return hist_concat

def plot_individual_histogram(hist, image_name, save_dir):
    """Plots and saves the concatenated histogram for a single image."""
    plt.figure(figsize=(10, 6))
    plt.plot(hist, color='blue')
    plt.xlabel('Concatenated HSV and LAB Bins')
    plt.ylabel('Probability')
    plt.tight_layout()

    hist_image_filename = os.path.join(save_dir, f'{image_name}_concatenated_histogram.png')
    plt.savefig(hist_image_filename)
    plt.close()

    print(f'Concatenated histogram for {image_name} saved as {hist_image_filename}')

def plot_combined_histograms(hist1, hist2, save_dir):
    # Plot both concatenated histograms on the same plot
    plt.figure(figsize=(10, 6))

    plt.plot(hist1, color='blue', label='Image 1')
    plt.plot(hist2, color='orange', label='Image 2')
    
    plt.xlabel('Concatenated HSV and LAB Bins')
    plt.ylabel('Probability')
    plt.legend()

    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    hist_image_filename = os.path.join(save_dir, 'combined_histograms.png')
    plt.tight_layout()
    plt.savefig(hist_image_filename)
    plt.close()

    print(f'Combined histograms saved as {hist_image_filename}')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <image_path1> <image_path2>")
        sys.exit(1)

    img_path1 = sys.argv[1]
    img_path2 = sys.argv[2]
    save_dir = 'histograms'

    # Compute concatenated histograms for both images
    hist_concat1 = compute_histograms(img_path1)
    hist_concat2 = compute_histograms(img_path2)

    # Extract the image names (without file extension) for labeling
    image_name1 = os.path.splitext(os.path.basename(img_path1))[0]
    image_name2 = os.path.splitext(os.path.basename(img_path2))[0]

    # Plot and save individual concatenated histograms
    plot_individual_histogram(hist_concat1, image_name1, save_dir)
    plot_individual_histogram(hist_concat2, image_name2, save_dir)

    # Plot and save the combined histograms
    plot_combined_histograms(hist_concat1, hist_concat2, save_dir)
