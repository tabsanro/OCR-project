import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 200)
engine.save_to_file("스캔 진행중",'run_OCR.mp3')
engine.runAndWait()