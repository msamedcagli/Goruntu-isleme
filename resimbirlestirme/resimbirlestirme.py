import cv2
import numpy as np

img1 = cv2.imread("img1.png")
img2 = cv2.imread("img2.jpeg")

img1 = cv2.resize(img1, (400, 400))
img2 = cv2.resize(img2, (400, 400))

hor = np.hstack((img1, img2))
cv2.imshow("Horizontal Image", hor)

ver = np.vstack((img1, img2))
cv2.imshow("Vertical Image", ver)

cv2.waitKey(0)
cv2.destroyAllWindows()
