import numpy as np
import cv2

cam = cv2.VideoCapture(0)
while True:
	ret, img = cam.read()
	cv2.imshow('cam', img)
	cv2.imwrite('check.jpg',img)
	ch = cv2.waitKey(5)
	if ch == 27:
		break
cv2.destroyAllWindows()
