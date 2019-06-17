import cv2
import os
img = cv2.imread('1.jpg', 1)
path = 'D:/OpenCV/Scripts/Images'
cv2.imwrite(os.path.join(path , 'waka.jpg'), img)
cv2.waitKey(0)
