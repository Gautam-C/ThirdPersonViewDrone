from djitellopy import tello
import numpy as np
import cv2
import time

fpsCounter = 0
start = int(round(time.time() * 1000))

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detection(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)    # bounding box


while cap.isOpened():
    _, frame = cap.read()

    detection(frame)
    fpsCounter+=1
    
    cv2.imshow('window', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    temp = int(round(time.time() * 1000))

    duration = (temp - start) / 1000

    if duration >= 30:
        print(f"{duration}s")
        print(f"{fpsCounter} frames")
        print(f"{fpsCounter/duration} fps")
        break


cap.release()
cv2.destroyAllWindows()