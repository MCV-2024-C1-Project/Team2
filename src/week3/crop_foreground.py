import cv2
import os


def crop_foreground_from_mask(image_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all .jpg images in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            # Construct the paths for the image and the corresponding masks
            image_path = os.path.join(image_folder, filename)
            base_name = os.path.splitext(filename)[0]
            mask1_path = os.path.join(image_folder, f"{base_name}_mask_s_contour1.png")
            mask2_path = os.path.join(image_folder, f"{base_name}_mask_s_contour2.png")

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading {filename}, skipping.")
                continue

            # Process the first mask if it exists
            if os.path.exists(mask1_path):
                crop_and_save_foreground(image, mask1_path, output_folder, base_name, 1)

            # Process the second mask if it exists
            if os.path.exists(mask2_path):
                crop_and_save_foreground(image, mask2_path, output_folder, base_name, 2)


def crop_and_save_foreground(image, mask_path, output_folder, base_name, contour_num):
    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error loading mask {mask_path}, skipping.")
        return

    # Find the bounding box of the white region (foreground) in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in {mask_path}, skipping.")
        return

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the image using the bounding box
    cropped_image = image[y:y+h, x:x+w]

    # Save the cropped image
    output_path = os.path.join(output_folder, f"{base_name}_cropped_contour{contour_num}.jpg")
    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped image saved to {output_path}")


# Example usage
image_folder = 'datasets/qst2_w3'
output_folder = 'src/week3/cropped_qst2_w3'
crop_foreground_from_mask(image_folder, output_folder)
