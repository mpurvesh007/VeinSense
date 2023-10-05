# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 22:17:20 2023

@author: Purvesh
"""

from skimage.metrics import structural_similarity
from matplotlib import pyplot as plt
import cv2
import numpy as np

# Function to calculate Intersection over Union (IoU)
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Load images
test_img = cv2.imread('C:\\Users\\Purvesh\\OneDrive\\Documents\\DISSERTATION\\images_db1\\.png')
gr_img = cv2.imread('C:\\Users\\Purvesh\\OneDrive\\Documents\\DISSERTATION\\annotated\\255.png')
pred = cv2.imread("C:\\Users\\Purvesh\\OneDrive\\Documents\\DISSERTATION\\Dilated_Image_255.png")

# Convert images to grayscale
gr_img_gray = cv2.cvtColor(gr_img, cv2.COLOR_BGR2GRAY)
pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)

# Calculate Structural Similarity Index (SSIM)
ssim = structural_similarity(gr_img_gray, pred_gray)

# Threshold the grayscale images to create binary masks
_, gr_mask = cv2.threshold(gr_img_gray, 1, 255, cv2.THRESH_BINARY)
_, pred_mask = cv2.threshold(pred_gray, 1, 255, cv2.THRESH_BINARY)

# Calculate IoU
iou = calculate_iou(gr_mask, pred_mask)

# Print SSIM and IoU scores
print("SSIM:", ssim)
print("IoU:", iou)

# Plot the images for visualization
plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('GrounTrurth')
plt.imshow(gr_img)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(pred)

plt.show()

# REFERENCES
#This file has the code which uses opencv library [1], matplotlib library [2], numpy library [3] and Scikit library [4].
#[1] Bradski, G., 2000. The OpenCV Library. Dr. Dobb27;s Journal of Software Tools.
#[2] J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp.
#[3] Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.
#[4] Van der Walt, S., Sch"onberger, Johannes L, Nunez-Iglesias, J., Boulogne, Franccois, Warner, J. D., Yager, N., … Yu, T. (2014). scikit-image: image processing in Python. PeerJ, 2, e453. 