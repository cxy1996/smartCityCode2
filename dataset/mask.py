import numpy as np
import cv2


circle = np.zeros((3000,4000), dtype="uint8")
cv2.circle(circle, (2000, 1500), 2000, 255, -1)
cv2.imwrite("circle.jpg", circle)