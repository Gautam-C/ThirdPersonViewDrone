import cv2
from djitellopy import Tello
import threading
import numpy as np

class WebcamVideoStream:
	def __init__(self, src=0, name="WebcamVideoStream"):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

		# initialize the thread name
		self.name = name

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		t = threading.Thread(target=self.update, name=self.name, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

#face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

lower_thresh = 200
minDist = 1000
pr1 = 255
pr2 = 10
minR = 5
maxR = 160

def back_detection(frame):
    green_frame = frame[:, :, 1]
    _, green_frame = cv2.threshold(green_frame, lower_thresh, 255, cv2.THRESH_TOZERO)

    green_frame = cv2.medianBlur(green_frame, 5)

    circles_temp = cv2.HoughCircles(green_frame, cv2.HOUGH_GRADIENT, 1, minDist, param1=pr1, param2=pr2, minRadius=minR, maxRadius=maxR)

    if circles_temp is None:
        final = [-1, -1, -1]
        return final

    circles = np.uint16(np.around(circles_temp))    # rounds values in circles_temp to nearest integer

    if len(circles[0]) > 1:
        print("error: more than one tracking target detected")
    elif len(circles[0]) != 1:
        print("error: no tracking targets found")
    
    cX = -1
    cY = -1
    cR = -1
    for (x, y, r) in circles[0]:
        cX = x
        cY = y
        cR = r
        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)    # draw bounding circle
        cv2.circle(frame, (x, y), 2, (255, 0, 0), 2)    # draw center of circle

    final = [cX, cY, cR]
    return final

#drone = Tello()

#drone.connect()
#drone.streamoff()
#drone.streamon()

#cap = cv2.VideoCapture(drone.get_udp_video_address())
# cap = cv2.VideoCapture("udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size=278877")

#drone_cam = WebcamVideoStream("udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size=278877", "drone_cam")
drone_cam = WebcamVideoStream(0, "drone cam")
drone_cam.start()
while not drone_cam.stopped:
    frame = drone_cam.read()

    if not drone_cam.grabbed:
        continue
    #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
    '''
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2) # bounding box
    '''
    coords = back_detection(frame)
    cv2.imshow('test', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

drone_cam.stop()
cv2.destroyAllWindows()
#drone.streamoff()