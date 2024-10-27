import os
import sys
import cv2
import numpy as np


image_path = 'cropped_qsd2_w3/'


old_stdout = sys.stdout
log_file = open("task1_output.log", "w", encoding='utf-8')
sys.stdout = log_file


def is_noisy(image):

    channels = cv2.split(image)  # This will give three channels: B, G, R

    laplacian_vars = []

    # Loop over each channel (B, G, R) and compute the Laplacian variance
    for channel in channels:
        laplacian = cv2.Laplacian(channel, cv2.CV_64F)
        laplacian_vars.append(laplacian.var())

    total_var = np.mean(laplacian_vars)

    return total_var > 4000


def compute_brightness(image):

    B, G, R = cv2.split(image)

    # Compute the brightness using the weighted sum
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    # brightness = 0.2126 * R + 0.7152 * G + 0.0722 * B

    # Compute the average brightness
    average_brightness = np.mean(brightness)

    return average_brightness


def get_image_index(filename):

    base_name = filename.replace('.jpg', '')
    index = int(base_name)

    return index


def multiply_hue(image, hue_factor):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    # Multiply the hue channel by the factor
    h = np.uint8((h.astype(np.float32) * hue_factor) %
                 180)  # Hue values wrap around at 180

    # Merge the channels back
    hsv_modified = cv2.merge([h, s, v])

    transformed_image = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)

    return transformed_image


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


hue_threshold = 100
for filename in os.listdir(image_path):
    if filename.lower().endswith(('.jpg')):

        image = cv2.imread(image_path + filename)
        # img_index = get_image_index(filename)

        average_brightness = compute_brightness(image)

        if is_noisy(image):

            best_result = None
            best_psnr = 0

            for h in range(10, 30, 5):
                filtered_image = cv2.fastNlMeansDenoisingColored(
                    image, None, h, 10, 7, 21)
                psnr = calculate_psnr(image, filtered_image)

                if psnr > best_psnr:
                    best_psnr = psnr
                    best_result = filtered_image

                    kernel = np.ones((3, 3), np.uint8)
                    best_result = cv2.morphologyEx(
                        filtered_image, cv2.MORPH_OPEN, kernel)

            cv2.imwrite('filtered_cropped_qsd2_w3/' + filename, best_result)

        elif average_brightness > hue_threshold:

            hue_factor = 1.5
            transformed_image = multiply_hue(image, hue_factor)

            cv2.imwrite('filtered_cropped_qsd2_w3/' + filename, transformed_image)

        else:
            cv2.imwrite('filtered_cropped_qsd2_w3/' + filename, image)

sys.stdout = old_stdout
log_file.close()
