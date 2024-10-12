import cv2


# Paths to the image and mask
image_path = 'datasets/qst2_w2/00003.jpg'
mask_path = 'results/week2/QST2/method1/00003.png'

# Load the original image
image = cv2.imread(image_path)

# Load the mask (ensure it's grayscale)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Ensure mask is binary (if needed)
_, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

# Find the contours of the mask to detect the bounding box of the foreground
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the bounding box coordinates of the largest contour (assuming the largest contour is the foreground)
if contours:
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the foreground area from the original image using the bounding box
    cropped_foreground = image[y:y+h, x:x+w]

    # Save the cropped foreground
    cv2.imwrite('cropped_foreground.png', cropped_foreground)

    # Optionally, display the cropped result
    cv2.imshow('Cropped Foreground', cropped_foreground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contours found in the mask.")
