import cv2 as cv
import numpy as np
from PIL import Image

img = Image.open("D:/image/malware/05e9e1a3f6675dcb173ba6dd8e27a8ee.png")
img = np.array(img)
img[:, :, 0] *= 0
img[:, :, 1] *= 0
# cv.imshow(mat=img, winname="show")
for i in range(0, 256):
    for j in range(0, 256):
        if img.item(i, j, 2) != 0 :
            print(str(i) + ", " + str(j) +", "+ str(img.item(i, j, 2)))


