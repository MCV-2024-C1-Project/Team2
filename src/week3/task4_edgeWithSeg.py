import cv2
import numpy as np
import os

# directory for the masks
directory = 'datasets/qsd2_w3'

# define function for enhancing the mask
import cv2
import numpy as np

# Define function for enhancing the mask
def contour_mask(mask):
    # Find contours in the input mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding rectangles for all contours
    rectangles = [cv2.boundingRect(contour) for contour in contours]
    # Sort rectangles by area in descending order
    rectangles = sorted(rectangles, key=lambda r: r[2] * r[3], reverse=True)

    height, width = mask.shape
    total_area = height * width

    # Find the two largest non-overlapping rectangles
    largest_rectangles = []
    for i, rect1 in enumerate(rectangles):
        x1, y1, w1, h1 = rect1
        # Check if this rectangle overlaps with any already selected rectangles
        is_overlapping = False
        for rect2 in largest_rectangles:
            x2, y2, w2, h2 = rect2
            # Check for overlap
            if not (x1 + w1 < x2 or x1 > x2 + w2 or y1 + h1 < y2 or y1 > y2 + h2):
                is_overlapping = True
                break

        # If it's not overlapping, add it to the list
        if not is_overlapping:
            largest_rectangles.append(rect1)

        # Stop if we have found two non-overlapping rectangles
        if len(largest_rectangles) == 2:
            break

    # Additional check for relative size difference between the two largest rectangles
    if len(largest_rectangles) == 2:
        # Calculate the area of the largest and second-largest rectangles
        area1 = largest_rectangles[0][2] * largest_rectangles[0][3]
        area2 = largest_rectangles[1][2] * largest_rectangles[1][3]

        # If the second-largest rectangle is significantly smaller than the largest one, keep only the largest
        if area2 < 0.2 * area1:  # Adjust the threshold (0.2) as needed
            largest_rectangles = [largest_rectangles[0]]

    # Create individual masks for each rectangle
    contour_masks = []
    for idx, rect in enumerate(largest_rectangles):
        x, y, w, h = rect
        # Create a blank mask for each rectangle
        individual_mask = np.zeros_like(mask)
        # Draw the rectangle on the individual mask
        cv2.rectangle(individual_mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)
        contour_masks.append(individual_mask)

    # Assign contour masks to return variables
    if len(contour_masks) == 2:
        return contour_masks[0], contour_masks[1]
    elif len(contour_masks) == 1:
        return contour_masks[0], None
    else:
        return None, None

   
#img = cv2.imread('datasets/qsd2_w3/00000_mask_s.png', cv2.IMREAD_GRAYSCALE)

#contour_mask(img)

# Iterate through the directory to process all .jpg files
for filename in os.listdir(directory):
    if filename.endswith('mask_s.png'):
        img_path = os.path.join(directory, filename)

        # Load the image
        img = cv2.imread(img_path)
        img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get contour masks using the modified function
        mask_1, mask_2 = contour_mask(img_greyscale)
        
        # Generate filenames for the contour masks
        base_name = os.path.splitext(filename)[0]
        mask_1_filename = f"{base_name}_contour1.png"
        mask_1_path = os.path.join(directory, mask_1_filename)

        # Save the first mask
        if mask_1 is not None:
            cv2.imwrite(mask_1_path, mask_1)
            print(f"Saved {mask_1_filename}.")
        
        # Save the second mask if it exists
        if mask_2 is not None:
            mask_2_filename = f"{base_name}_contour2.png"
            mask_2_path = os.path.join(directory, mask_2_filename)
            cv2.imwrite(mask_2_path, mask_2)
            print(f"Saved {mask_2_filename}.")
            
print('Finished processing the data folder')



