import cv2
import imutils
import numpy as np
import mediapipe as mp
from easyocr import Reader
import pyttsx3
import pygame
from imutils.perspective import four_point_transform
from urllib.request import urlopen
import time


class Button:
    def __init__(self, x, y, w=50, h=50, sound=None):
        self.x0 = x
        self.y0 = y
        self.w = w
        self.h = h
        self.onclick = 0
        self.sound = sound
        self.color = (0, 255, 0)

    def _on_click(self):
        if self.onclick == 1:
            return 0
        pygame.mixer.stop()
        self.onclick = 1
        self.sound.play()
        self.color = (255, 0, 0)

    def check(self, x, y):
        if self.x0-10 < x < self.x0+10 + self.w and self.y0+10 < y < self.y0+20 + self.h:
            self._on_click()
        else:
            self.onclick = 0
            self.color = (0, 255, 0)


def make_scan_image(image, width, ksize=(5, 5), min_threshold=75, max_threshold=200):
    image_list_title = []
    image_list = []

    org_image = image.copy()  # 이미지복사 (깊은복사)
    image = imutils.resize(image, width=width)  # 크기 변환
    ratio = org_image.shape[1] / float(image.shape[1])  # 비율 저장

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백
    blurred = cv2.GaussianBlur(gray, ksize, 0)  # 블러링
    edged = cv2.Canny(blurred, min_threshold, max_threshold)  # 외곽선 추출

    # contours를 찾아 크기순으로 정렬
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    findCnt = None

    # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
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
        transform_image = org_image, 0, 0

    return transform_image

def run_ocr(image):
    print("[INFO] OCR'ing input image...")
    sound = pygame.mixer.Sound("run_OCR.mp3")
    sound.play()
    langs = ['ko', 'en']
    reader = Reader(lang_list=langs, gpu=True)
    '''
    reader = Reader(['ko'], gpu=True,
                    model_storage_directory='model',  # 학습모델위치
                    user_network_directory='model',  # yaml 위치
                    recog_network='custom')
    '''
    img = image.copy()
    results = reader.readtext(img)
    box = []
    i = 0
    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

        if text:
            engine.save_to_file(text, 'text.mp3')
            engine.runAndWait()
            sound = pygame.mixer.Sound("text.mp3")
            globals()['b_{}'.format(i)] = Button(tl[0], tl[1], tr[0] - tl[0], bl[1] - tl[1], sound=sound)
            box.append(globals()['b_{}'.format(i)])
            i += 1

        # 추출한 영역에 사각형을 그리고 인식한 글자를 표기합니다.
        cv2.rectangle(img, tl, br, (0, 255, 0), 1)
        # business_card_image = putText(business_card_image, text, tl[0], tl[1] - 60, (0, 255, 0), 50)
        cv2.putText(img, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    # cv2.imshow('result', img)

    return results, box

pygame.init()

### 제스처 부분
max_num_hands = 1
gesture = {
    0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'rock', 8: 'spiderman', 9: 'yeah', 10: 'ok', 11:'scale'
}
rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# tts 엔진
engine = pyttsx3.init()
engine.setProperty('rate', 200)

box = []
p_x, p_y = 0, 0
on_11 = 0
scale_ratio = 1

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_hand = img.copy()
    img_hand = cv2.cvtColor(img_hand, cv2.COLOR_BGR2RGB)

    hand_result = hands.process(img_hand)

    h, w, _ = img_hand.shape
    fingerx, fingery = 0, 0
    if hand_result.multi_hand_landmarks is not None:
        for res in hand_result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v = v2 - v1  # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            angle = np.degrees(angle)  # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # Draw gesture result
            if idx == 1:
                fingerx = int(res.landmark[8].x * w)
                fingery = int(res.landmark[8].y * h)
                #cv2.putText(img_hand, text="({0}, {1}".format(fingerx, fingery), org=(30, 30),
                            #fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                cv2.line(img_hand, (fingerx, fingery), (fingerx, fingery), (255, 255, 255), 5)
            elif idx == 0:
                pygame.mixer.stop()

            if idx == 11:
                thumb = res.landmark[4]
                index = res.landmark[8]
                if on_11 == 0:
                    start = time.time()
                    diff0 = ((thumb.x-index.x)**2 +(thumb.y-index.y)**2)**(1/2)
                    on_11 = 1
                else:
                    diff1 = ((thumb.x-index.x)**2 +(thumb.y-index.y)**2)**(1/2)
                    end = time.time()
                    if end - start > 0.08:
                        diff = int((diff1-diff0)*200)
                        if diff >= 7:
                            scale_ratio += 0.1
                        elif diff <= -7:
                            scale_ratio -= 0.1
                        on_11 = 0
            else:
                on_11 = 0


    for b0 in box:
        b0.check(fingerx, fingery)  # onclick 되어 있는지 확인
        cv2.rectangle(img_hand, (b0.x0, b0.y0), (b0.x0 + b0.w, b0.y0 + b0.h), b0.color, 2)  # 사각형 그리기

    img_print = cv2.cvtColor(img_hand, cv2.COLOR_RGB2BGR)
    img_scale = cv2.resize(img_print, (int(w * scale_ratio), int(h * scale_ratio)), \
                           interpolation=cv2.INTER_AREA)
    cv2.imshow('hand', img_scale)

    keycode = cv2.waitKey(1)
    if keycode == ord('a'):
        try:
            img_crop, p_x, p_y = make_scan_image(img, width=200, ksize=(5, 5), min_threshold=20, max_threshold=100)
            ocr_result, box = run_ocr(img_crop)
            for b0 in box:
                b0.x0 += p_x
                b0.y0 += p_y
            ocr = 1
            print("OCR is done")
            print(p_x, p_y)
        except:
            print("OCR is failed")
            pass

    elif keycode == ord('r'):
        if ocr == 1:
            if idx == 1:
                i = 0
                for (bbox, text, prob) in ocr_result:
                    (tl, tr, br, bl) = bbox
                    if (tl[0] + p_x < fingerx < tr[0] + p_x and tl[1] + p_y < fingery < bl[1] + p_y):
                        try:
                            print(text)
                            engine.say(text)
                            engine.runAndWait()
                        except:
                            pass
                    i += 1

    if keycode == ord('q'):
        break