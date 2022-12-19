from easyocr import Reader
import cv2
from PIL import Image
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
        self.sound.stop()
        self.onclick = 1
        self.sound.play()
        self.color = (255,0,0)
    def check(self, x, y):
        if self.x0 < x < self.x0 + self.w and self.y0 < y < self.y0 + self.h:
            self._on_click()
        else:
            self.onclick = 0
            self.color = (0, 255, 0)

pygame.init()

langs = ['ko', 'en']
reader = Reader(lang_list=langs, gpu=True)

engine = pyttsx3.init()
engine.setProperty('rate', 125)
img = Image.open('capture.jpg')
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
