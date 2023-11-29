import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def getLabel(centroids, cx, cy):
    distances = [math.sqrt((x - cx)**2 + (y - cy)**2) for x, y in centroids]
    closest_label = distances.index(min(distances)) + 1
    return closest_label

centroids = [
    [63, 85], [148, 85], [236, 85], [330, 85],
    [63, 250], [148, 250], [236, 250], [330, 250],
    [63, 428], [148, 428], [236, 428], [330, 428],
    [410, 428], [506, 428], [594, 428], [690, 428]
]

img0 = mpimg.imread('Taller8/Imagen0.jpg')
img1 = mpimg.imread('Taller8/Imagen3.jpg')
gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
subtraction = gray0 - gray1
_, binarization = cv2.threshold(subtraction, 1, 250, cv2.THRESH_BINARY)
kernel = np.ones((3, 3), np.uint8)
dilation = cv2.dilate(binarization, kernel, iterations=1)
erosion = cv2.erode(binarization, kernel, iterations=2)

# Opening operation to remove small noise
opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)

# Paint the holes that are not in the original image in blue
blue_mask = np.zeros_like(img1, dtype=np.uint8)
blue_mask[:, :] = [255, 0, 0]
result_blue = cv2.bitwise_and(blue_mask, blue_mask, mask=binarization)

# Identify and write a number and draw a filled blue box only in the blue squares representing missing chips
font = cv2.FONT_HERSHEY_SIMPLEX
chip_number = 1

# Convert the resulting image to grayscale
result_gray = cv2.cvtColor(result_blue, cv2.COLOR_BGR2GRAY)

# Find contours in the resulting image
contours, _ = cv2.findContours(result_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate the total number of chips
total_chips = len(centroids)

for contour in contours:
    # Calculate the area of the contour
    area = cv2.contourArea(contour)
    # Set an area threshold to identify missing chips
    if area > 200:
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        # Draw a filled blue box in the original image
        cv2.rectangle(img1, (x, y), (x+w, y+h), (255, 0, 0), thickness=cv2.FILLED)
        # Calculate the position to place the number
        text_position = (x + w // 2 - 10, y + h // 2 + 5)
        # Calculate the correct number of the missing chip
        chip_number = getLabel(centroids, x + w // 2, y + h // 2)
        # Write the number on the original image
        cv2.putText(img1, str(chip_number), text_position, font, 0.5, (255, 255, 255), 2)

# Show the resulting image
cv2.imshow("image", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
