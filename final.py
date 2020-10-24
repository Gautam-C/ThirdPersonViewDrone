import queue
import threading
from djitellopy import Tello
import cv2
import numpy as np
from queue import Queue


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



# drone = tello.Tello()

# upper_body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')

front = False
on = True

lower_thresh = 200
minDist = 1000
pr1 = 255
pr2 = 10
minR = 0
maxR = 160

global_coords = [-1, -1, 3]

# cap = cv2.VideoCapture('udp://'+drone.tello_ip+':11111')
# cap = drone.get_video_capture()
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size=50000000")
# drone_cam = WebcamVideoStream("udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size=278877", "drone cam")
# drone_cam.start()

def upper_body_detection(frame):
    
    # detect upper bodies
    # upper_bodies = upper_body_cascade.detectMultiScale(gray_frame)    # too resource intensive
    
    green_frame = frame[:, :, 1]

    _, green_frame = cv2.threshold(green_frame, lower_thresh, 255, cv2.THRESH_TOZERO)
    
    green_frame = cv2.medianBlur(green_frame, 5)

    cv2.imshow('thresh+blur', green_frame)

    # detect tracking target(s)
    circles_temp = cv2.HoughCircles(green_frame, cv2.HOUGH_GRADIENT, 1, minDist, param1=pr1, param2=pr2, minRadius=minR, maxRadius=maxR)

    if circles_temp is None:
        final = [-1, -1, -1]
        return final

    circles = np.uint16(np.around(circles_temp))    # rounds values in circles_temp to nearest integer

    if len(circles[0]) > 1:
        print("error, more than one tracking target detected")

    cX = -1
    cY = -1
    cR = -1
    for (x, y, r) in circles[0]:
        cX = x
        cY = y
        cR = r
        cv2.circle(frame, (x, y), r, (0, 0, 255), 2) # tracking circle
        cv2.circle(frame, (x, y), 2, (255, 0, 0), 2) # tracking circle center

    '''
    # finds closest upper body to tracking target
    index = -1
    distance = 1469
    for ind, (x, y, w, h) in enumerate(upper_bodies):
        dist = math.sqrt(((x + (w/2)) - cX)**2 + ((y + (h/2)) - cY)**2) # calculates distance between center of upper bodies and tracking target
        
        # checks if distance from current upper body to tracking target is less than previous smallest upper body distance
        if dist <= distance:
            distance = dist
            index = ind
    
    (rX, rY, rW, rH) = upper_bodies[index]  # store the co-ords of rectangle closest to tracking target
    cv2.rectangle(frame, (rX, rY), (rX + rW, rY + rH), (0, 255, 0), 2)  # draw bounding box
    cv2.circle(frame, (rX + (rW/2), rY + (rH/2)), 2, (255, 0, 0), 2)    # draw center of bounding box
    '''
    # final = [[rX, rY, rW, rH, [cX, cY]]   includes upper-body back_detection
    final = [cX, cY, cR]
    return final

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

def movement(circle_coords, drone):
    x, y, r = circle_coords

    if x == -1:
        return

    if not front:

        if r < 20:
            drone.move_forward(20)
        elif r > 100:
            drone.move_back(20)
    else:
        if r < 20:
            drone.move_back(20)
        elif r > 100:
            drone.move_forward(20)

    if x > 510:
        drone.move_right(20)
    elif x < 450:
        drone.move_left(20)
    
    if y > 390:
        drone.move_down(20)
    elif y < 330:
        drone.move_up(20)

def front_movement(circle_coords):
    x, y, r = circle_coords

    if x == -1:
        return

    if not front:

        if r < 20:
            drone.move_forward(20)
        elif r > 100:
            drone.move_back(20)
    else:
        if r < 20:
            drone.move_back(20)
        elif r > 100:
            drone.move_forward(20)

    if x > 670:
        drone.move_left(20)
    elif x < 610:
        drone.move_right(20)
    
    if y > 390:
        drone.move_up(20)
    elif y < 330:
        drone.move_down(20)

def detection(drone_memory, drone_cam_memory, drone_detect_memory):
    
    drone_array = np.ndarray((1,), dtype=np.object, buffer=drone_memory.buf)

    d_cam_array = np.ndarray((1,), dtype=np.object, buffer=drone_cam_memory.buf)

    drone_detect_array = np.ndarray((1,), dtype=np.bool, buffer=drone_detect_memory.buf)

    while not d_cam_array[0].stopped:

        if not drone_detect_array[0]:
            continue

        frame = d_cam_array[0].read()

        # if drone_cam frame is dropped skip cycle
        if not d_cam_array[0].grabbed:
            continue

        cv2.imshow('camera', frame)
        coords = back_detection(frame)
        print(coords)
        movement(coords, drone_array[0])

        if cv2.waitKey(1) & 0xFF == 27:
            drone_memory.close()
            drone_cam_memory.close()
            drone_detect_memory.close()
            break

def view_camera(drone_memory, drone_cam_memory, drone_detect_memory):
    drone_array = np.ndarray((1,), dtype=np.object, buffer=drone_memory.buf)

    d_cam_array = np.ndarray((1,), dtype=np.object, buffer=drone_cam_memory.buf)

    drone_detect_array = np.ndarray((1,), dtype=np.bool, buffer=drone_detect_memory.buf)

    while not d_cam_array[0].stopped:
        
        frame = d_cam_array[0].read()
        
        # if drone_cam frame is dropped skip cycle
        if not d_cam_array[0].grabbed:
            continue

        cv2.imshow('camera', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            cv2.destroyAllWindows()
            cv2.waitKey(1) & 0xFF
            drone_cam_memory.close()
            drone_detect_memory.close()
            drone_memory.close()
            break
        elif key == ord('s'):
            drone_detect_array[0] = False
            if front:
                drone_array[0].curve_xyz_speed(0, 100, 0, 100, 50, 0, 5)
                drone_array[0].rotate_counter_clockwise(180)
            else:
                drone_array[0].curve_xyz_speed(0, -100, 0, 100, -50, 0, 5)
                drone_array[0].rotate_clockwise(180)
            drone_detect_array[0] = True

def view_camera_t():

    while not drone_cam.stopped:
        
        frame = drone_cam.read()
        
        # if drone_cam frame is dropped skip cycle
        if not drone_cam.grabbed:
            continue

        cv2.imshow('camera', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == ord('s'):
            if front:
                drone.curve_xyz_speed(0, 100, 0, 100, 50, 0, 5)
                drone.rotate_counter_clockwise(180)
            else:
                drone.curve_xyz_speed(0, -100, 0, 100, -50, 0, 5)
                drone.rotate_clockwise(180)

def detection_t(drone_cam, drone):

    while not drone_cam.stopped:

        frame = drone_cam.read()

        # if drone_cam frame is dropped skip cycle
        if not drone_cam.grabbed:
            continue

        coords = back_detection(frame)
        global_coords = coords
        print(coords)
        movement(coords, drone)
'''
def key_check():
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        on = False
    elif key == ord('s'):
        if front:
            drone.curve_xyz_speed(0, 100, 0, 100, 50, 0, 5)
        else:
            drone.curve_xyz_speed(0, -100, 0, 100, -50, 0, 5)
'''

# START PROGRAM
if __name__ == "__main__":

    drone = Tello()
    drone.connect()
    print(drone.get_battery())
    drone.streamoff()
    drone.streamon()

    coords_queue = Queue(1)

    drone_cam = WebcamVideoStream("udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size=278877", "drone cam")
    #drone_cam = WebcamVideoStream(0, "drone cam")
    drone_cam.start()

    print(f"{drone_cam.stream.get(cv2.CAP_PROP_FRAME_WIDTH)} x {drone_cam.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    '''
    smm = SharedMemoryManager()
    smm.start()

    drone_array = np.array([drone,])
    drone_memory = smm.SharedMemory(size=drone_array.nbytes)
    drone_memory_buf = np.ndarray(drone_array.shape, dtype=drone_array.dtype, buffer=drone_memory.buf)
    drone_memory_buf[:] = drone_array[:]

    drone_cam_array = np.array([drone_cam,])
    drone_cam_memory = smm.SharedMemory(size=drone_cam_array.nbytes)
    drone_cam_memory_buf = np.ndarray(drone_cam_array.shape, dtype=drone_cam_array.dtype, buffer=drone_cam_memory.buf)
    drone_cam_memory_buf[:] = drone_cam_array[:]

    drone_detect_on_array = np.array([True,])
    drone_detect_on_memory = smm.SharedMemory(size=drone_detect_on_array.nbytes)
    drone_detect_on_buf = np.ndarray(drone_detect_on_array.shape, dtype=drone_detect_on_array.dtype, buffer=drone_detect_on_memory.buf)
    drone_detect_on_buf[:] = drone_detect_on_array[:]
    '''
    

    '''
    while not drone_cam_memory_buf[0].stopped:
        
        frame = drone_cam_memory_buf[0].read()
        # if drone_cam frame is dropped skip cycle
        if not drone_cam_memory_buf[0].grabbed:
            continue

        cv2.imshow("press c to start", frame)

        key_start = cv2.waitKey(1) & 0xFF
        if key_start == ord("c"):
            cv2.destroyWindow("press c to start") # remove start window
            cv2.waitKey(1) & 0xFF   # fix window not dissappearing
            break
    '''
    '''
    drone_cam_memory.close()    # close main instance of SharedMemory name="drone_cam_object"

    drone_memory_buf[0].takeoff()
    drone_memory_buf[0].move_up(50)   # get drone to body level

    drone_memory.close()
    '''

    drone.takeoff()
    drone.move_up(50)

    #vid_process = multiprocessing.Process(target=view_camera, args=[drone_memory, drone_cam_memory, drone_detect_on_memory])
    #detect_process = multiprocessing.Process(target=detection, args=[drone_memory, drone_cam_memory, drone_detect_on_memory])

    # vid_thread = threading.Thread(target=view_camera)
    detect_thread = threading.Thread(target=detection_t, args=[drone_cam, drone])

    #vid_process.start()
    #detect_process.start()

    # vid_thread.start()
    detect_thread.start()

    while not drone_cam.stopped:
        frame = drone_cam.read()

        if not drone_cam.grabbed:
            continue

        if global_coords[0] == -1:
            global_coords[0] = 0
            global_coords[1] = 0
            global_coords[2] = 2

        cv2.circle(frame, (global_coords[0], global_coords[1]), global_coords[2], (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    drone.land()
    drone_cam.stop()
    detect_thread.join()

    #vid_process.join()  # waits for video process to be killed
    #detect_process.terminate()  # then kills detection process

    #vid_thread.join()
    #detect_thread.join()
    '''
    final_drone_cam_memory = shared_memory.SharedMemory(name="drone_cam_object")
    final_drone_cam_array = np.ndarray((1,), dtype=np.object, buffer=final_drone_cam_memory.buf)    # load camera in
    final_drone_cam_array[0].stop() # stop camera
    final_drone_cam_memory.close()  # close camera storage
    final_drone_cam_memory.unlink() # delete camera storage

    final_drone_detect_on_memory = shared_memory.SharedMemory(name="detect_on") # load detect bool storage in
    final_drone_detect_on_memory.close()    # close bool storage
    final_drone_detect_on_memory.unlink()   # delete bool storage 

    final_drone_memory = shared_memory.SharedMemory(name="drone_object")
    final_drone_memory = np.ndarray((1,), dtype=np.object, buffer=final_drone_memory.buf) # load drone storage in
    final_drone_memory[0].land()    # land drone
    final_drone_memory[0].streamoff()   # turn off stream
    final_drone_memory.close()  # close drone storage
    final_drone_memory.unlink() # delete drone storage

    smm.shutdown()
    '''
    '''
    if not front:
        rear_circle_coords = back_detection(frame)
        print(rear_circle_coords)
        rear_movement(rear_circle_coords)
        
    elif front:
        front_circle_coords = upper_body_detection(frame)
        print(front_circle_coords)
        front_movement(front_circle_coords)
    '''

    '''
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == ord('s'):
        front = not front
        if front:
            drone.curve_xyz_speed(0, 100, 0, 100, 50, 0, 5)
        else:
            drone.curve_xyz_speed(0, -100, 0, 100, -50, 0, 5)
    '''

    cv2.destroyAllWindows()
    drone.streamoff()