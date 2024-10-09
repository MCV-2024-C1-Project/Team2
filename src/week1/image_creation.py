
import cv2
import os

# Load the image in RGB using OpenCV
image_path = 'qsd1_w1/00000.jpg'  # Replace with the correct path to your image
output_directory = 'examples/'  # Directory to save the images

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

rgb_image = cv2.imread(image_path)  # OpenCV loads images as BGR by default

# Check if the image is loaded correctly
if rgb_image is None:
    print(f"Error: Could not load the image at {image_path}. Please check the file path or file integrity.")
else:
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Convert the RGB image to grayscale
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # Convert the RGB image to HSV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Convert the RGB image to LAB
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)

    # Convert the RGB image to YCrCb and extract chrominance (Cr, Cb channels)
    ycrcb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    chrominance_image = ycrcb_image[:, :, 1:]  # Extract Cr and Cb channels only (ignoring Y)

    # Convert back to BGR format before saving the RGB image
    rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Save the images as JPG
    cv2.imwrite(os.path.join(output_directory, 'rgb_image.jpg'), rgb_bgr)
    cv2.imwrite(os.path.join(output_directory, 'gray_image.jpg'), gray_image)
    cv2.imwrite(os.path.join(output_directory, 'hsv_image.jpg'), hsv_image)
    cv2.imwrite(os.path.join(output_directory, 'lab_image.jpg'), lab_image)
    cv2.imwrite(os.path.join(output_directory, 'chrominance_image.jpg'), chrominance_image)

    print(f"Images saved successfully to {output_directory}")
