import cv2
from skimage.segmentation import clear_border, chan_vese
import numpy as np


img = cv2.imread('datasets/qsd2_w3/00011.jpg')

cv2.imshow('image', img); cv2.waitKey(0); cv2.destroyAllWindows()

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('image', img_hsv); cv2.waitKey(0); cv2.destroyAllWindows()

img_h = img_hsv[:,:,0]

cv2.imshow('image', img_h); cv2.waitKey(0); cv2.destroyAllWindows()

img_s = img_hsv[:,:,1]

cv2.imshow('image', img_s); cv2.waitKey(0); cv2.destroyAllWindows()

claeh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_s_clahe = claeh.apply(img_s)

cv2.imshow('image', img_s_clahe); cv2.waitKey(0); cv2.destroyAllWindows()

cv = chan_vese(img_s_clahe, mu=0.08, lambda1=0.8, lambda2=1.2, tol=1e-3, max_num_iter=300,
                       dt=0.5, init_level_set="checkerboard", extended_output=True)

# Convert the result to uint8 to display it
cv_uint8 = (cv[0] * 255).astype(np.uint8)

cv2.imshow('image', cv_uint8); cv2.waitKey(0); cv2.destroyAllWindows()

mask = clear_border(cv_uint8)

cv2.imshow('image', mask); cv2.waitKey(0); cv2.destroyAllWindows()