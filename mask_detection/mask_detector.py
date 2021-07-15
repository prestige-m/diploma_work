import time
import cv2
import os
import imutils
import numpy as np

from tensorflow.keras.models import load_model

from face_identify import FaceRecognition
from mask_detection import FaceDetector

model = load_model('mask_recognition_v3.h5')
#model = load_model('model_v1.h5')

label = {
    0: {"name": "Mask only in the chin", "color": (51, 153, 255), "id": 0, "ratio": 10.0},
    1: {"name": "Mask below the nose", "color": (255, 255, 0), "id": 1, "ratio": 4.0},
    2: {"name": "Without mask", "color": (0, 0, 255), "id": 2, "ratio": 0},
    3: {"name": "With mask ok", "color": (0, 102, 51), "id": 3, "ratio": 1.85},
}

video_capture = cv2.VideoCapture(0)
time.sleep(2.0)
detector = FaceDetector()

while True:
    cut_face = None
    # Capture frame-by-frame
    _, frame = video_capture.read()
    frame_copy = frame.copy()
    #frame = imutils.resize(frame, width=400)

    faces_list = []
    preds = []
    faces = detector.extract_faces(frame)

    for face_box in faces:
        box = face_box['rect']
        face_frame = face_box['face_frame']
        conf = face_box['confidence']
        x, y, w, h = box[0], box[1], box[2], box[3]

        # -----
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray_face = gray[y:y + h + 50, x:x + w + 50]
        # gray_face = cv2.resize(gray_face, (300, 300))
        # gray_face = gray_face / 255
        # gray_face = np.expand_dims(gray_face, axis=0)
        # gray_face = gray_face.reshape((1, 300, 300, 1))

        amm = model.predict(face_frame)[0]
        #prediction = int(np.argmax(model.predict(face_frame, batch_size=32)))
        prediction = model.predict(face_frame, batch_size=32)
        pred_index = int(np.argmax(prediction, axis=1))

        recognizer = FaceRecognition()
        cut_face = recognizer.crop_mask(frame_copy, box, label[pred_index]["ratio"])


        classification = label[pred_index]["name"]
        color = label[pred_index]["color"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, label[pred_index]["id"])
        cv2.putText(frame, classification, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{len(faces)} detected face", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
                    cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cut_face is not None:
        cv2.imshow('Video22', cut_face)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()