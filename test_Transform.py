import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


img = cv2.imread("capture.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백
blurred = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX = 0)  # 블러링
thresh = cv2.adaptiveThreshold(
    blurred,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)
#thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)

plt.imshow(thresh, cmap='gray')
plt.show()
_, contours, _ = cv2.findContours(
    thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)
'''

# contours를 찾아 크기순으로 정렬
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

findCnt = None

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
    if len(approx) == 4 and cv2.contourArea(c) > 500:
        findCnt = approx
        break

if findCnt is not None:
    a = findCnt[0][0][0] + findCnt[0][0][1]
    b = findCnt[1][0][0] + findCnt[1][0][1]
    if a < b:
        x = findCnt[0][0][0] * ratio
        y = findCnt[0][0][1] * ratio
    else:
        x = findCnt[1][0][0] * ratio
        y = findCnt[1][0][1] * ratio
    x = int(x)
    y = int(y)
    transform_image = four_point_transform(org_image, findCnt.reshape(4, 2) * ratio), x, y
else:
    transform_image = image, 0, 0
    '''