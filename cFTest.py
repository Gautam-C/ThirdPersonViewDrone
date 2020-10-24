import concurrent.futures
import time
import cv2

def detect(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

    for (x, y, w, h)  in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return 'finished'


if __name__ == '__main__':

    start = time.perf_counter()

    img = cv2.imread('opencv-logo-white.png')

    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        for i in range(10):
            future1 = executor.submit(detect, img)
            futures.append(future1)

        for future in futures:
            print(future.result())

    finish = time.perf_counter()

    print(f"Finished in {round(finish - start, 2)}")