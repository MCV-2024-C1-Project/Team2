import cv2
import numpy as np
import os

# directory for the masks
directory = 'datasets/qsd2_w3'

# define function for enhancing the mask
def contour_mask(mask):

    # Apply Canny edge detection
    #edges = cv2.Canny(mask, threshold1=25, threshold2=50)
    # Dilate edges to close gaps
    #edges_dilated = cv2.dilate(edges, None, iterations=1)
    # Invert the image to make the foreground white
    #mask = cv2.bitwise_not(edges_dilated)
    #cv2.imshow('edges', mask); cv2.waitKey(0); cv2.destroyAllWindows()
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get bounding rectangles for all contours
    rectangles = [cv2.boundingRect(contour) for contour in contours]
    # Sort rectangles by area in descending order
    rectangles = sorted(rectangles, key=lambda r: r[2] * r[3], reverse=True)

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
    # Create a blank mask to draw the rectangles
    contour_mask = np.zeros_like(mask)

    # Draw the rectangles on the mask
    for rect in largest_rectangles:
        x, y, w, h = rect
        cv2.rectangle(contour_mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)

     # show result
    #cv2.imshow('edges', contour_mask); cv2.waitKey(0); cv2.destroyAllWindows()
    return mask, contour_mask
   
#img = cv2.imread('datasets/qsd2_w3/00000_mask_s.png', cv2.IMREAD_GRAYSCALE)

#contour_mask(img)

# Iterate through the directory to process all .jpg files
for filename in os.listdir(directory):
    if filename.endswith('mask_s.png'):
        img_path = os.path.join(directory, filename)

        # Load the image
        img = cv2.imread(img_path)
        img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask, contour = contour_mask(img_greyscale)
         # Save both masks as PNG files
        mask_final_filename = os.path.splitext(filename)[0] + '_fmask.png'
        mask_contours_filename = os.path.splitext(filename)[0] + '_contours.png'

        mask_final_path = os.path.join(directory, mask_final_filename)
        mask_contours_path = os.path.join(directory, mask_contours_filename)

        # Save the mask images
        cv2.imwrite(mask_final_path, mask)
        cv2.imwrite(mask_contours_path, contour)

        print(f"Saved {mask_final_filename} and {mask_contours_filename}.")
            
print('Finish the data folder proccessing')



