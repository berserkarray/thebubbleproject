# Perform PCA on the grayscale image
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib import pyplot

image = cv2.imread('thebubblesproject/vid1/ezgif-frame-001.jpg', 0)

def pca_background_subtraction(image):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # perform PCA on flattened gray image
    gray_2d = gray.reshape(-1, 1)
    mean, eig_vec = cv2.PCACompute(gray_2d, mean=None)
    projected = np.dot(gray_2d - mean, eig_vec)
    # reshape projected image back to 2D
    projected_2d = projected.reshape(gray.shape)
    # normalize and scale to [0, 255]
    cv2.normalize(projected_2d, projected_2d, 0, 255, cv2.NORM_MINMAX)
    projected_2d = np.uint8(projected_2d)
    # perform binary thresholding
    _, thresholded = cv2.threshold(projected_2d, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresholded

def bubble_detection(image):
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresholded = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(thresholded, kernel, iterations=1)
    # Convert the mask to grayscale with the bubbles represented as white pixels
    gray_mask = 255 - mask
    thresh = cv2.threshold(gray_mask, 200, 255, cv2.THRESH_BINARY)[1]
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours and label bubbles with serial numbers
    for i, contour in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 50 and cv2.contourArea(contour)<100:
            cv2.rectangle(gray_mask, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(gray_mask, str(i+1), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    return gray_mask

thresholded = pca_background_subtraction(image)
gray_mask = bubble_detection(thresholded)

# Display image
imS = cv2.resize(gray_mask, (720, 960))
cv2.imshow('image', imS)
cv2.waitKey(0)
cv2.destroyAllWindows()