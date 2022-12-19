import cv2
import numpy as np
from urllib.request import urlopen

class App:
    def __init__(self):
        self.buffer = b''
        #esp-cap stream
        url = 'http://192.168.4.1/' #Your url
        self.stream = urlopen(url)
print('no')
scan = App()
print('no')

print(3)

while True:
    scan.buffer += scan.stream.read(2560)
    head = scan.buffer.find(b'\xff\xd8')
    end = scan.buffer.find(b'\xff\xd9')
    try:
        if head > -1 and end > -1:
            #촬영 데이타를 jpg로 변환하기
            jpg = scan.buffer[head:end+2]
            scan.buffer = scan.buffer[end+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            cv2.imshow('test', img)
    except:
        pass
    keycode = cv2.waitKey(1)
    if keycode == ord('a'):
        cv2.imwrite('capture' + ".jpg", img)