import time
import cv2
import os
import imutils
import numpy as np

from tensorflow.keras.models import load_model
from mask_detection import FaceDetector

model = load_model('model/mask_recognition.h5')

label = {
    0: {"name": "Mask only in the chin", "color": (51, 153, 255), "id": 0},
    1: {"name": "Mask below the nose", "color": (255, 255, 0), "id": 1},
    2: {"name": "Without mask", "color": (0, 0, 255), "id": 2},
    3: {"name": "With mask ok", "color": (0, 102, 51), "id": 3},
}

video_capture = cv2.VideoCapture(0)
time.sleep(2.0)
detector = FaceDetector()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    faces_list = []
    preds = []
    faces = detector.extract_faces(frame)

    for face_box in faces:
        box = face_box['rect']
        face_frame = face_box['face_frame']
        conf = face_box['confidence']
        x, y, w, h = box[0], box[1], box[2], box[3]

        pred = int(np.argmax(model.predict(face_frame)))
        classification = label[pred]["name"]
        color = label[pred]["color"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, label[pred]["id"])
        cv2.putText(frame, classification, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{len(faces)} detected face", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
                    cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()