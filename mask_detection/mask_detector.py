import time
import cv2
import os
import imutils
import numpy as np

from tensorflow.keras.models import load_model

from face_identify import FaceRecognition
from mask_detection import FaceDetector
from deepface.commons import distance as dst

model = load_model('mask_recognition_v3.h5')
#model = load_model('model_v1.h5')

# label = {
#     0: {"name": "Mask only in the chin", "color": (51, 153, 255), "id": 0, "ratio": 10.0},
#     1: {"name": "Mask below the nose", "color": (255, 255, 0), "id": 1, "ratio": 4.0},
#     2: {"name": "Without mask", "color": (0, 0, 255), "id": 2, "ratio": 0},
#     3: {"name": "With mask ok", "color": (0, 102, 51), "id": 3, "ratio": 1.85},
# }

mask_labels = {
    0: {"name": "Mask only in the chin", "color": (51, 153, 255), "id": 0, "ratio": 15.0},
    1: {"name": "Mask below the nose", "color": (255, 255, 0), "id": 1, "ratio": 9.0},
    2: {"name": "Without mask", "color": (0, 0, 255), "id": 2, "ratio": 0},
    3: {"name": "With mask ok", "color": (0, 102, 51), "id": 3, "ratio": 2.5},
}

video_capture = cv2.VideoCapture(0)
time.sleep(2.0)
detector = FaceDetector()
#SIZE = (224, 224)
SIZE = (160, 160)
MODEL_NAME = 'Facenet'
CONFIDENSE = 0.85

def euc_l2(img1_representation, img2_representation):
    return dst.findEuclideanDistance(dst.l2_normalize(img1_representation),
                              dst.l2_normalize(img2_representation))

distance_metrics = {'cosine': dst.findCosineDistance, 'euclidean':dst.findEuclideanDistance, 'euclidean_l2': euc_l2}

while True:
    cut_face = None
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()
    #frame = imutils.resize(frame, width=400)

    faces_list = []
    preds = []
    faces = detector.extract_faces(frame)
    recognizer = FaceRecognition()

    encs, labels = recognizer.dataset['arr_0'], recognizer.dataset['arr_1']

    all_data = {}
    for label, enc in zip(labels, encs):
        if all_data.get(label) is None:
            all_data[label] = [enc]
        else:
            all_data[label].append(enc)

    for face_box in faces:
        box = face_box['rect']
        face_frame = face_box['face_frame']
        conf = face_box['confidence']
        #x, y, w, h = box[0] - 30, box[1] - 35, box[2] + 30, box[3] + 20
        x, y, w, h = box[0], box[1], box[2], box[3]
        new_box = (x, y, w, h)

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

        cut_face = recognizer.crop_mask(frame_copy, new_box, mask_labels[pred_index]["ratio"])
        #if cut_face is not None:

        start_x, end_x = x, x + w
        start_y, end_y = y, y + h
        cut_face2 = frame[start_y:end_y, start_x:end_x]

        new_box2 = start_y, end_x, end_y, start_x


        classification = mask_labels[pred_index]["name"]
        color = mask_labels[pred_index]["color"]
        if cut_face is not None:
            #face_array = recognition.align_face(face_array, face_keypoints)
            face_array = recognizer.preprocess_face(cut_face, target_size=SIZE)


            detection_info = recognizer.identify_face(face_array, probability_level=0.5)

            new_data2 = {}
            for label, embeddings in all_data.items():

                label_dist2 = []

                for source_embedding in embeddings:
                    for distance_metric, func in distance_metrics.items():
                        dist = func(source_embedding, detection_info['embeddings'])
                        threshold = dst.findThreshold(MODEL_NAME, distance_metric)
                        label_dist2.append(dist <= threshold)

                new_data2[label] = sum(label_dist2) / len(label_dist2)

            class_new = max(new_data2, key=new_data2.get)
            conf = new_data2[class_new]

            cv2.putText(frame,
                        f" - {class_new} - {conf}",
                        (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            cv2.putText(frame,f"{classification} - {detection_info['name']} - {round(detection_info['probability'] * 100, 2)}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)



        cv2.rectangle(frame, (x, y), (x + w, y + h), color, mask_labels[pred_index]["id"])
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