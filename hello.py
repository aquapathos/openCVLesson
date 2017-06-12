import cv2
import numpy as np
a = np.zeros((300,300))
b = np.ones((150,150))
a[0:150,0:150]=b
a[150:300,150:300]=b
cv2.imshow("A",a)
cv2.waitKey(0)
