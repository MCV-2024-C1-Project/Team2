import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


def plot_histograms_separately(img_path, save_dir):
    # Grayscale histogram
    img_grey = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hist_grey = cv2.calcHist([img_grey], [0], None, [256], [0, 256])
    hist_grey /= hist_grey.sum()

    # RGB Histograms
    img_BGR = cv2.imread(img_path)
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    hist_r = cv2.calcHist([img_RGB], [0], None, [256], [0, 256])
    hist_r /= hist_r.sum()

    hist_g = cv2.calcHist([img_RGB], [1], None, [256], [0, 256])
    hist_g /= hist_g.sum()

    hist_b = cv2.calcHist([img_RGB], [2], None, [256], [0, 256])
    hist_b /= hist_b.sum()

    # LAB Histograms
    img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
    hist_l = cv2.calcHist([img_LAB], [0], None, [256], [0, 256])
    hist_l /= hist_l.sum()

    hist_a = cv2.calcHist([img_LAB], [1], None, [256], [0, 256])
    hist_a /= hist_a.sum()

    hist_b_lab = cv2.calcHist([img_LAB], [2], None, [256], [0, 256])
    hist_b_lab /= hist_b_lab.sum()

    # HSV Histograms
    img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([img_HSV], [0], None, [256], [0, 256])
    hist_h /= hist_h.sum()

    hist_s = cv2.calcHist([img_HSV], [1], None, [256], [0, 256])
    hist_s /= hist_s.sum()

    hist_v = cv2.calcHist([img_HSV], [2], None, [256], [0, 256])
    hist_v /= hist_v.sum()

    # Save each histogram as a separate image
    os.makedirs(save_dir, exist_ok=True)

    # Plot Grayscale histogram
    plt.figure()
    plt.plot(hist_grey, color='black')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Probability')
    plt.xlim([0, 256])
    plt.tight_layout()
    hist_image_filename = os.path.join(save_dir, 'grayscale_histogram.png')
    plt.savefig(hist_image_filename)
    plt.close()
    print(f'Grayscale histogram saved as {hist_image_filename}')

    # Plot RGB histograms
    plt.figure()
    plt.plot(hist_r, color='red', label='Red Channel')
    plt.plot(hist_g, color='green', label='Green Channel')
    plt.plot(hist_b, color='blue', label='Blue Channel')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Probability')
    plt.xlim([0, 256])
    plt.legend()
    plt.tight_layout()
    hist_image_filename = os.path.join(save_dir, 'rgb_histogram.png')
    plt.savefig(hist_image_filename)
    plt.close()
    print(f'RGB histogram saved as {hist_image_filename}')

    # Plot LAB histograms
    plt.figure()
    plt.plot(hist_l, color='black', label='L Channel')
    plt.plot(hist_a, color='red', label='A Channel')
    plt.plot(hist_b_lab, color='blue', label='B Channel')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Probability')
    plt.xlim([0, 256])
    plt.legend()
    plt.tight_layout()
    hist_image_filename = os.path.join(save_dir, 'lab_histogram.png')
    plt.savefig(hist_image_filename)
    plt.close()
    print(f'LAB histogram saved as {hist_image_filename}')

    # Plot HSV histograms
    plt.figure()
    plt.plot(hist_h, color='orange', label='Hue Channel')
    plt.plot(hist_s, color='green', label='Saturation Channel')
    plt.plot(hist_v, color='blue', label='Value Channel')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Probability')
    plt.xlim([0, 256])
    plt.legend()
    plt.tight_layout()
    hist_image_filename = os.path.join(save_dir, 'hsv_histogram.png')
    plt.savefig(hist_image_filename)
    plt.close()
    print(f'HSV histogram saved as {hist_image_filename}')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    save_dir = 'histograms'

    plot_histograms_separately(img_path, save_dir)
