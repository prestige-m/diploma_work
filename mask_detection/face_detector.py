from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os

class FaceDetector:

    CONFIDENCE_LEVEL = 0.5
    facenet_model = None

    def __init__(self, model_directory: str = "model"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prototxt_path = os.path.sep.join([current_dir, model_directory, "deploy.prototxt"])
        weights_path = os.path.sep.join([current_dir, model_directory, "res10_300x300_ssd_iter_140000.caffemodel"])
        self.facenet_model = cv2.dnn.readNet(prototxt_path, weights_path)

    def extract_faces(self, frame):
        h, w = frame.shape[:2]
        mean_value = (104.0, 177.0, 123.0)
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), mean_value)

        # pass the blob through the network and obtain the face detections
        self.facenet_model.setInput(blob)
        detections = self.facenet_model.forward()

        face_boxes = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.CONFIDENCE_LEVEL:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                filter_gap = 20
                if (w - endX <= filter_gap or h - endY <= filter_gap or startX <= filter_gap or startY <= filter_gap) or (
                        startX > endX or startY > endY):
                    continue

                # extract the face ROI
                face_frame = frame[startY:endY, startX:endX]
                face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                face_frame = cv2.resize(face_frame, (224, 224))
                face_frame = img_to_array(face_frame)
                face_frame = preprocess_input(face_frame)
                face_frame = np.expand_dims(face_frame, axis=0)

                face_boxes.append({'face_frame': face_frame, 'rect': (startX, startY, endX - startX, endY - startY),
                                    'confidence': confidence})

        return face_boxes
