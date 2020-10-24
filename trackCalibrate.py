from djitellopy import tello
import cv2
import numpy as np

#drone1 = Tello()

#drone1.streamon()

#cap = cv2.VideoCapture(drone1.get_udp_video_address())

import time

fpsCounter = 0
trials = []
start = int(round(time.time() * 1000))

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("test.mp4")

def detection(frame):
    green_frame = frame[:, :, 1]
    _, green_frame = cv2.threshold(green_frame, 200, 255, cv2.THRESH_TOZERO)

    circles = cv2.HoughCircles(green_frame, cv2.HOUGH_GRADIENT, 1, 2000, param1=50, param2=30, minRadius=20, maxRadius=0)
    if circles is None:
        return
    
    circles = np.uint16(np.around(circles))

    if len(circles[0]) > 1:
        print("error: more than one tracking target detected")
    
    for (x, y, r) in circles[0]:
        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 2, (255, 0, 0), 2)

while cap.isOpened():
    _, frame = cap.read()

    detection(frame)
    fpsCounter += 1

    cv2.imshow('window', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    temp = int(round(time.time() * 1000))
    duration = (temp - start) / 1000

    if duration >= 30:
        if len(trials) >= 10:
            break
        nextAddition = [duration, fpsCounter, fpsCounter/duration]
        trials.append(nextAddition)

        print(f"{duration}s")
        print(f"{fpsCounter} frames")
        print(f"{fpsCounter/duration} fps")
        fpsCounter = 0
        start = int(round(time.time() * 1000))
        

cap.release()
cv2.destroyAllWindows()
tim_total = -1
frames_total = -1
fps_total = -1.0
for (tim, frames, fps) in trials:
    tim_total += tim
    frames_total += frames
    fps_total += fps

print(f"{tim_total}s")
print(f"{frames_total} frames")
print(f"{fps_total} fps")