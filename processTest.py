from multiprocessing import shared_memory
import multiprocessing
from multiprocessing.managers import SharedMemoryManager
import time
import cv2
import numpy as np
import sys

from numpy.core.arrayprint import dtype_is_implied

class Rand:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def toString(self):
        print(f"{self.name} : {self.age}")

def do_something(num):
    print(f"changing to {num}")
    cap_memory = shared_memory.SharedMemory(name="cap_mem")
    cap_buf = cap_memory.buf
    cap_buf[0] = num
    print(f"do_something : {cap_memory.buf[0]}")
    print(f"done changing to {num}")
    cap_memory.close()

def do():
    cap_memory = shared_memory.SharedMemory(name="cap_mem")
    cap_memory.buf[0] = 3
    print("changed to 3")


def key_quit():
    cap_mem = shared_memory.SharedMemory(name="cap_mem")
    cap = cap_mem.buf[0]

    while cap.isOpened():
        _, frame = cap.read()
        if not _:
            continue
        cv2.imshow('quit', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cap_mem.close()
            break



def detect(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

    for (x, y, w, h)  in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


start = time.perf_counter()

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    cap_array = np.array([cap,])
    smm = SharedMemoryManager()
    smm.start()
    cap_memory = 

    #p1 = multiprocessing.Process(target=do_something, args=[2])
    p2 = multiprocessing.Process(target=key_quit)

    #p1.start()
    p2.start()

    #p1.join()
    #p2.terminate()
    p2.join()

    final_memory = shared_memory.SharedMemory(name="cap_mem")

    print(f'final : {final_memory.buf[0]}')
    final_memory.close()
    final_memory.unlink()
    
    finish = time.perf_counter()

    print(f"Finished in {round(finish - start, 2)}")