import threading
from djitellopy import Tello
import cv2
import threading

#drone = tello.Tello()
drone = Tello()

drone.connect()
drone.streamoff()
drone.streamon()

cap = cv2.VideoCapture("udp://@0.0.0.0:11111")

def vid_stream():
    while cap.isOpened():
        _, frame = cap.read()

        if frame is None:
            continue

        cv2.imshow("drone", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cap.release()
    cv2.destroyALlWindows()
    drone.streamoff()

def move():
    drone.takeoff()
    drone.move_forward(20)
    drone.move_back(20)

    drone.move_right(20)
    drone.move_left(20)

    drone.move_back(20)
    drone.move_forward(20)

    drone.move_left(20)
    drone.move_right(20)
    drone.land()
'''
vid_thread = threading.Thread(target=vid_stream)
move_thread = threading.Thread(target=move)

move_thread.start()
vid_thread.start()

move_thread.join()
vid_thread.join()
'''
move()