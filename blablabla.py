import pyttsx3
import time
import cv2
import pygame



pygame.init()


engine = pyttsx3.init()
engine.setProperty('rate',400)
engine.say('아시발존나좆같네')
engine.runAndWait()