import os
import sys
import cv2
import numpy as np

image_path = 'datasets/qsd1_w3/'


old_stdout = sys.stdout
log_file = open("task1_output.log", "w", encoding='utf-8')
sys.stdout = log_file


def is_noisy(image, threshold=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    return std_dev > threshold


def apply_filters(image):
    if is_noisy(image):
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
    else:
        denoised = cv2.medianBlur(image, 5)  # Less aggressive filtering

    return denoised


def calculate_psnr(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    if mse == 0:
        return 100

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


for filename in os.listdir(image_path):
    if filename.lower().endswith(('.jpg')):
        image = cv2.imread(image_path + filename)

        if is_noisy(image):
            print("Image is noisy, applying filters.")

            best_result = None
            best_psnr = 0

            for h in range(10, 50, 2):
                filtered_image = cv2.fastNlMeansDenoisingColored(
                    image, None, h, 10, 7, 21)
                psnr = calculate_psnr(image, filtered_image)

                if psnr > best_psnr:
                    best_psnr = psnr
                    best_result = filtered_image

                    kernel = np.ones((3, 3), np.uint8)
                    best_result = cv2.morphologyEx(
                        filtered_image, cv2.MORPH_OPEN, kernel)

            cv2.imwrite('filtered_img/' + filename, best_result)

        else:
            print("Image is not noisy, no filtering applied.")

sys.stdout = old_stdout
log_file.close()
