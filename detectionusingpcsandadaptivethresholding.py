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

def edge_linking(binary_image):
    # Create a copy of the binary image to store the linked edges
    linked_image = np.zeros_like(binary_image)

    # Find the starting point of an edge (the first white pixel)
    start_point = np.argwhere(binary_image == 255)[0]
    current_point = start_point

    # Set the current direction to 0 (right direction)
    current_direction = 0

    while True:
        # Mark the current point as part of the linked edge
        linked_image[current_point[0], current_point[1]] = 255

        # Check the neighbors in the current direction
        neighbor_indices = [
            (current_point[0] - 1, current_point[1]),  # Up
            (current_point[0] - 1, current_point[1] + 1),  # Up-right
            (current_point[0], current_point[1] + 1),  # Right
            (current_point[0] + 1, current_point[1] + 1),  # Down-right
            (current_point[0] + 1, current_point[1]),  # Down
            (current_point[0] + 1, current_point[1] - 1),  # Down-left
            (current_point[0], current_point[1] - 1),  # Left
            (current_point[0] - 1, current_point[1] - 1)  # Up-left
        ]

        # Find the first neighbor that is part of the edge
        for i in range(current_direction, current_direction + 8):
            neighbor_index = i % 8
            neighbor = neighbor_indices[neighbor_index]
            if binary_image[neighbor[0], neighbor[1]] == 255:
                # Set the current point and direction to the found neighbor
                current_point = neighbor
                current_direction = (neighbor_index + 5) % 8
                break
        else:
            # If no edge pixel is found in the neighbors, the edge is complete
            break

    return linked_image

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
    # Apply Canny edge detection
    edges = cv2.Canny(thresh, 100, 200)

    # Perform morphological operations to enhance the edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    # Find contours
    contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours and label bubbles with serial numbers
    for i, contour in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 30 and cv2.contourArea(contour)<50:
            cv2.rectangle(gray_mask, (x,y), (x+w,y+h), (0,255,0), 1)
            cv2.putText(gray_mask, str(i+1), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    return gray_mask

thresholded = pca_background_subtraction(image)
gray_mask = bubble_detection(thresholded)

# Display image
# imS = cv2.resize(gray_mask, (720, 960))
cv2.imwrite('image.jpg', gray_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()