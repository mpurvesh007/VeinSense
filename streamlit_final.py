# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:19:28 2023

@author: Purvesh
"""
import io
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from tensorflow import keras
import time
import base64
from io import BytesIO
import os
from PIL import Image


def image_processing(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get a binary image
    threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 141, 7)
    inverted_image = 255 - threshold_image

    # Apply morphological operations (erosion and dilation) for noise removal and border clearing
    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(inverted_image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

    # Inverse the image to get the black objects on a white background
    final_binary_image = 255 - dilated_image
    cv2.imwrite("final_binary.jpg",final_binary_image)
    # Convert the binary image to RGB and highlight the veins in red
    vein_highlight_rgb = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2BGR)
    vein_highlight_rgb[np.where((vein_highlight_rgb == [255, 255, 255]).all(axis=2))] = [0, 0, 255]

    # Overlay the highlighted veins on the original image with Gaussian blur for smoothing
    blurred_image = cv2.GaussianBlur(vein_highlight_rgb, (5, 5), 0)
    smoothed_image = cv2.addWeighted(blurred_image, 1.5, vein_highlight_rgb, -0.5, 0)

    # Convert the original image and the smoothed veins image to HSV color space
    original_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    vein_mask_hsv = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2HSV)

    # Replace the hue and saturation of the original image with those of the smoothed veins image
    original_hsv[..., 0] = vein_mask_hsv[..., 0]
    original_hsv[..., 1] = vein_mask_hsv[..., 1] * 0.6
    final_output_image = cv2.cvtColor(original_hsv, cv2.COLOR_HSV2BGR)
    return final_output_image


def predict(image_1):
    model = keras.models.load_model("vein_test(B)_34.hdf5", compile=False)
    image_1 = cv2.resize(image_1, (256, 256))
    img = cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR)
    img_1 = np.expand_dims(img, axis=0)

    # Make predictions using the loaded model
    predict_1 = model.predict(img_1)
    predict_2 = predict_1[0, :, :, 0]

    # Normalize the "predict_2" image to the range [0, 255] (uint8) for displaying
    predict_2 = (predict_2 * 255).astype(np.uint8)

    # Resize the "predict_2" image to the original image's size
    predict_2_resized = cv2.resize(predict_2, (image_1.shape[1], image_1.shape[0]))

    _, thresh = cv2.threshold(predict_2_resized, 127, 255, cv2.THRESH_BINARY)

    backtorgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    backtorgb[np.where((backtorgb == [255, 255, 255]).all(axis=2))] = [0, 0, 255]

    # Overlaying the images
    blur = cv2.GaussianBlur(backtorgb, (5, 5), 0)
    smooth = cv2.addWeighted(blur, 1.5, backtorgb, -0.5, 0)
    img_hsv = cv2.cvtColor(image_1, cv2.COLOR_BGR2HSV)
    color_mask_hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * 0.6
    img_masked = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    return img_masked


def main():
    st.sidebar.title("VeinSense: Deep Learning-Powered Dorsal Vein Recognition")

    # Add a description for the app
    st.sidebar.markdown(
        "Discover Vein Highlighter: By employing cutting-edge models such as VGG and ResNet our technology expertly analyses dorsal hand images with pinpoint precision. Using carefully curated dataset training these models ensure accurate vein segmentation for seamless visualization and analysis through our app for medical imaging research studies - offering insights in medical imaging research studies as a whole! Compare output from image processing with outputs from deep neural networks for an overall view."
    )

    # Create an "Upload File" button
    uploaded_files = st.sidebar.file_uploader(
        "***Upload JPG or PNG image***", type=["jpg", "png"], accept_multiple_files=True
    )

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_name = os.path.splitext(file_name)[0]

            # Convert the uploaded image to a PIL image
            pil_image = Image.open(uploaded_file)
            img = np.asarray(pil_image)
            resized_image = cv2.resize(img, (256, 256))
            col1, col2, col3 = st.columns(3)
            col1.image(resized_image, caption="Uploaded Image", use_column_width=True)

            # Add a progress bar while the image is being processed
            progress_text = "Processing the image..."
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.05)  # Simulating a small delay for processing
                progress_bar.progress(percent_complete + 1)

            # Perform the prediction on the uploaded image
            ip_output_image = image_processing(img)
            ip_output_image = cv2.resize(ip_output_image, (256, 256))

            output_image = predict(resized_image)

            output_image = cv2.resize(output_image, (256, 256))
            # Use st.beta_columns to display images side by side

            # Display the predicted result in the second column

            col2.image(
                ip_output_image,
                caption="Predicted Result Using Image Processing",
                use_column_width=True,
            )

            col3.image(
                output_image,
                caption="Predicted Result Using Deep Neural Network",
                use_column_width=True,
            )

            concatenated_image = Image.fromarray(
                np.concatenate(
                    [
                        np.array(resized_image),
                        np.array(ip_output_image),
                        np.array(output_image),
                    ],
                    axis=1,
                )
            )
            concatenated_image_bytes = io.BytesIO()
            concatenated_image.save(concatenated_image_bytes, format="JPEG")
            concatenated_image_bytes.seek(0)

            # Download concatenated image
            st.download_button(
                "Download Concatenated Image",
                data=concatenated_image_bytes,
                file_name=f"concatenated_image_{file_name}.jpg",
            )

            # Clear the progress bar after displaying the predicted result
            progress_bar.empty()


if __name__ == "__main__":
    main()
