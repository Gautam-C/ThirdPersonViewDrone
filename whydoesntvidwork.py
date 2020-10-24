import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    
    if not _:
        continue
    
    cv2.imshow('test', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        break