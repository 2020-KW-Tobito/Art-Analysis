import cv2 as cv
import numpy as np
import os
import cv2


datadir = "./Sketch_Dataset/"

files = os.listdir(datadir)
categories=[]

for i in files:
    categories.append(i)

print(categories)
num_classes=len(categories)

directory = "Sketch_distancetransform/"

for c in range(240):
    file = categories[c]
    png_file = datadir + file

    gray = cv2.imread(png_file, 0)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    dist_transform = cv.distanceTransform(thresh, cv.DIST_L2, 5)
    # dist_transform  함수를 사용하면 실수 타입(float32)의 이미지가 생성됩니다. 화면에 보여주려면 normalize 함수를 사용해야 합니다.
    result = cv.normalize(dist_transform, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
    filename = directory + file
    cv2.imwrite(filename, result)

