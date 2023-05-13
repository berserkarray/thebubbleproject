import cv2

# Load image
image = cv2.imread('thebubblesproject/vid1/ezgif-frame-001.jpg')

blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Convert to grayscale
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

# Apply thresholding to remove noise
thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and label bubbles with serial numbers
for i, contour in enumerate(contours):
    (x,y,w,h) = cv2.boundingRect(contour)
    if cv2.contourArea(contour) > 50 and cv2.contourArea(contour)<100: # ignore small contours (adjust threshold as needed)
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(image, str(i+1), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

# Display image
imS = cv2.resize(image, (720, 960))
cv2.imshow('image', imS)
cv2.waitKey(0)
cv2.destroyAllWindows()