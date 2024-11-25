# Face Recognition

# Importing the libraries
import cv2

# Load Haar Cascade files for face and eye detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Write 'Face' above the rectangle
        text_position = (x, y - 10)  # Position above the rectangle
        cv2.putText(frame, 'Face', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            # Write 'Face' above the rectangle
            text_position = (ex, ey - 10)  # Position above the rectangle
            cv2.putText(roi_color, 'Eye', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()