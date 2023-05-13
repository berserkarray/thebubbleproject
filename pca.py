import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# Load the image
image = cv2.imread('thebubblesproject/vid1/ezgif-frame-001.jpg', 0)

# Perform PCA on the grayscale image
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
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(thresholded, kernel, iterations=1)
    return mask

def bubble_clustering(bubbles, eps=20, min_samples=5):
    # Get the indices of the detected bubbles
    y_indices, x_indices = np.where(bubbles == 255)
    points = np.column_stack((x_indices, y_indices))
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    # Get the labels assigned to each point by the clustering algorithm
    labels = clustering.labels_
    # Count the number of clusters (ignoring the -1 label assigned to noise points)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # Initialize an array to store the masks of each cluster
    masks = [np.zeros_like(bubbles) for i in range(num_clusters)]
    # Assign each point to its corresponding cluster mask
    for i in range(len(points)):
        if labels[i] != -1:
            masks[labels[i]][y_indices[i], x_indices[i]] = 255
    
    return masks

thresholded = pca_background_subtraction(image)
mask = bubble_detection(thresholded)
labels = bubble_clustering(mask)
imS = cv2.resize(mask, (540, 960))
cv2.imshow('obtained after running bubbles clustering', imS)
cv2.waitKey(0)
cv2.destroyAllWindows()