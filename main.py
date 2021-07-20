from mtcnn import MTCNN
import cv2
import face_recognition

detector = MTCNN()
# Load a videopip TensorFlow
video_capture = cv2.VideoCapture(0)

while (True):
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    #frame = cv2.resize(frame, (600, 400))
    boxes = face_recognition.face_locations(frame)
    #####rect.top(), rect.right(), rect.bottom(), rect.left()
    #boxes = detector.detect_faces(frame)

    conf = 1
    if boxes:

        # box = boxes[0]['box']
        # conf = boxes[0]['confidence']
        box = boxes[0]
        x, y, w, h = box[3], box[0], box[1], box[2]

        #x, y, w, h = box[0], box[1], box[2], box[3]

        if conf > 0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video_capture.release()