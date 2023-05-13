# thebubbleproject
## detectbubbles.py 
tried detecting bubbles by smoothing out image using the gaussian kernel of 5x5. then used a simple binary threshold followed by Adaptive thresholding used for further reduction in noise and highlighting the boundaries of bubbles. found contours using findContours method provided by cv2 api. considered only contours within 50-100 units of area. <br>
problems identified - findcontour not able to detect bubbles with subtle differece in dynamic range. reflection on tube adds to the false positive.
## detectionusingpcaandadaptivethresholding.py 
as the name suggests, used the approach from the paper. Performed PCA and then used adaptivethresholding. increase in false contours observed due to lack of edge linking. same detection approach as used in detectbubbles.py
## onlypcaanddetection.py
detectbubbles.py + pca, not sure why i did it but thought it might work. very bad result.
