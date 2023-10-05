# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 21:19:48 2023

@author: Purvesh
"""

import matplotlib.pyplot as plt

# Data for the scores
image_numbers = list(range(1, 16))
ssim_scores = [0.814233499, 0.817563608, 0.820071886, 0.805495959, 0.754964996, 0.760097216, 0.762441681, 0.800396214, 0.801164219, 0.799180833, 0.825147219, 0.819402199, 0.729315348, 0.734520358, 0.831065556]

# New x-axis labels
new_labels = [1, 8, 15, 20, 25, 106, 111, 118, 125, 129, 133, 136, 233, 238, 255]

# Plotting SSIM scores (Scatter Plot with Connected Line)
plt.figure(figsize=(10, 6))
plt.plot(image_numbers, ssim_scores, color='b', marker='o', label='SSIM Scores (Line)')
plt.scatter(image_numbers, ssim_scores, color='b', marker='o', label='SSIM Scores (Scatter)')
plt.xlabel('Image Number')
plt.ylabel('SSIM Score')
plt.title('SSIM Scores of Dorsal Hand Images')
plt.grid(True, axis='both', linestyle='--', alpha=0.7)
plt.legend()

# Setting custom x-axis labels
plt.xticks(image_numbers, new_labels) # Customize x-axis labels with new_labels

plt.tight_layout()
plt.show()

# Plotting IoU scores (Scatter Plot with Connected Line)
iou_scores = [0.229310666, 0.210993304, 0.186548302, 0.177245894, 0.117638822, 0.218267293, 0.20076398, 0.162219337, 0.149716096, 0.197922338, 0.114431474, 0.114192195, 0.263769321, 0.330029307, 0.045501743]

plt.figure(figsize=(10, 6))
plt.plot(image_numbers, iou_scores, color='r', marker='o', label='IoU Scores (Line)')
plt.scatter(image_numbers, iou_scores, color='r', marker='o', label='IoU Scores (Scatter)')
plt.xlabel('Image Number')
plt.ylabel('IoU Score')
plt.title('IoU Scores of Dorsal Hand Images')
plt.grid(True, axis='both', linestyle='--', alpha=0.7)
plt.legend()

# Setting custom x-axis labels
plt.xticks(image_numbers, new_labels) # Customize x-axis labels with new_labels

plt.tight_layout()
plt.show()

# REFERENCE
# This file has the code which uses matplotlib library [1] to plot the graphs 4:10 and 4:11 IN THE REPORT
#[1] citing J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp.