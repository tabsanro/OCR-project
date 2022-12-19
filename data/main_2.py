from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt
from pytesseract import Output
from easyocr import Reader
import pytesseract
import mediapipe as mp
import imutils
import cv2
import re
import requests
import numpy as np

pytesseract.pytesseract.tesseract_cmd = R'C:\Program Files\Tesseract-OCR\tesseract' #ocr엔진

def plt_imshow(title='image', img=None, figsize=(8,5)):
    plt.figure(figsize=figsize)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
             titles = []

             for i in range(len(img)):
                 titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i +1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()

def make_scan_image(image, width, ksize=(5,5), min_threshold=75, max_threshold=200):
    image_list_title = []
    image_list = []

    org_image = image.copy() #이미지복사 (깊은복사)
    image = imutils.resize(image, width=width) #크기 변환
    ratio = org_image.shape[1] / float(image.shape[1]) #비율 저장

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #흑백
    blurred = cv2.GaussianBlur(gray, ksize, 0) #블러링
    edged = cv2.Canny(blurred, min_threshold, max_threshold) #외곽선 추출

    # contours를 찾아 크기순으로 정렬
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    findCnt = None

    #정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
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
            x = findCnt[0][0][0]
            y = findCnt[0][0][1]
        else:
            x = findCnt[1][0][0]
            y = findCnt[1][0][1]
        transform_image = four_point_transform(org_image, findCnt.reshape(4, 2) * ratio), x, y
    else:
        fail = cv2.imread('img/img.png')
        transform_image = fail, 0, 0

    return transform_image

def run_ocr(image):
    print("[INFO] OCR'ing input image...")
    reader = Reader(lang_list=langs, gpu=True)
    img = image.copy()
    results = reader.readtext(img)
    for (bbox, text, prob) in results:
        print("[INFO] {:.4f}: {}".format(prob, text))

        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

        # 추출한 영역에 사각형을 그리고 인식한 글자를 표기합니다.
        cv2.rectangle(img, tl, br, (0, 255, 0), 1)
        # business_card_image = putText(business_card_image, text, tl[0], tl[1] - 60, (0, 255, 0), 50)
        cv2.putText(img, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('result', img)

    return results

langs = ['ko', 'en']

cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('cap', img)

    keycode = cv2.waitKey(1)
    if keycode == ord('a'):
        img_crop, x, y = make_scan_image(img, width=200, ksize=(5, 5), min_threshold=20, max_threshold=100)
        results = run_ocr(img_crop)

    elif keycode == ord('q'):
        break