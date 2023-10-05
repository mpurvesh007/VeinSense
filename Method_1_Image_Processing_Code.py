# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:30:32 2023

@author: Purvesh
"""

import cv2
import numpy as np

# Read the image
original_image = cv2.imread('C:\\Users\\Purvesh\\OneDrive\\Documents\\DISSERTATION\\images_db1\\233.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to get a binary image
threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 141, 7)
inverted_image = 255 - threshold_image

# Apply morphological operations (erosion and dilation) for noise removal and border clearing
kernel = np.ones((5, 5), np.uint8)
eroded_image = cv2.erode(inverted_image, kernel, iterations=1)
dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

# Inverse the image to get the black objects on a white background
final_binary_image = 255 - dilated_image

# Convert the binary image to RGB and highlight the veins in red
vein_highlight_rgb = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2RGB)
vein_highlight_rgb[np.where((vein_highlight_rgb == [255, 255, 255]).all(axis=2))] = [0, 0, 255]

# Overlay the highlighted veins on the original image with Gaussian blur for smoothing
blurred_image = cv2.GaussianBlur(vein_highlight_rgb, (5, 5), 0)
smoothed_image = cv2.addWeighted(blurred_image, 1.5, vein_highlight_rgb, -0.5, 0)

# Convert the original image and the smoothed veins image to HSV color space
original_hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
vein_mask_hsv = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2HSV)

# Replace the hue and saturation of the original image with those of the smoothed veins image
original_hsv[..., 0] = vein_mask_hsv[..., 0]
original_hsv[..., 1] = vein_mask_hsv[..., 1] * 0.6
final_output_image = cv2.cvtColor(original_hsv, cv2.COLOR_HSV2BGR)

# Display the final result with highlighted veins
cv2.imshow('Original Image', original_image)
cv2.imshow('Dilated Image', dilated_image)
cv2.imshow('Vein Highlighted Image', final_output_image)


# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()


#REFERENCES
# This file has the code which uses opencv library [1] and numpy library [60]. OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library [2]. This file gives output of all figures from 4:2 to 4:9 IN THE REPORT
#[1] Bradski, G., 2000. The OpenCV Library. Dr. Dobb27;s Journal of Software Tools.
#[2]Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.