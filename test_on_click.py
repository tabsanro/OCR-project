import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import pygame

class Button:
    def __init__(self, x, y, w=50, h=50, sound=None):
        self.x0 = x
        self.y0 = y
        self.w = w
        self.h = h
        self.onclick = 0
        self.sound = sound
        self.color = (0,255,0)
    def _on_click(self):
        if self.onclick == 1:
            return 0
        pygame.mixer.stop()
        self.onclick = 1
        self.sound.play()
        self.color = (255,0,0)
    def check(self, x, y):
        if self.x0-10 < x < self.x0+10 + self.w and self.y0-10 < y < self.y0+10 + self.h:
            self._on_click()
        else:
            self.onclick = 0
            self.color = (0, 255, 0)




pygame.init()

### 제스처 부분
max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok', 11:'scale'
}
rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

engine = pyttsx3.init()
engine.setProperty('rate', 125)
engine.save_to_file("라이라이차차차 부라보 부라보 해병대",'test.mp3')
engine.runAndWait()
sound = pygame.mixer.Sound("test.mp3")
sound2 = pygame.mixer.Sound("text.mp3")

box = []
b1 = Button(50, 50, 100,100, sound=sound)
b2 = Button(150, 50, sound=sound2)
b3 = Button(250, 50, sound=sound)
b4 = Button(350, 50, sound=sound)
box.append(b1)
box.append(b2)
box.append(b3)
box.append(b4)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img_hand = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
                cv2.putText(img_hand, text="({0}, {1}".format(fingerx, fingery), org=(30, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                cv2.line(img_hand, (fingerx,fingery), (fingerx,fingery), (255,255,255), 5)
            elif idx == 0:
                cv2.putText(img_hand, text="stop", org=(30, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                pygame.mixer.stop()
    for b0 in box:
        b0.check(fingerx, fingery) #onclick 되어 있는지 확인
        cv2.rectangle(img_hand, (b0.x0,b0.y0), (b0.x0+b0.w,b0.y0+b0.h), b0.color, 2) #사각형 그리기

    img_print = cv2.cvtColor(img_hand, cv2.COLOR_RGB2BGR)

    cv2.imshow('test', img_print)

    keycode = cv2.waitKey(1)
    if keycode == ord('q'):
        break