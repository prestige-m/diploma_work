# import math
#
# import imutils
# from imutils import paths
#
# import cv2
# import os
# import numpy as np
#
#
# # loading the keras facenet model
# #from keras.models import load_model
# from tensorflow.keras.models import load_model
#
# from matplotlib import pyplot as plt
# from matplotlib.patches import Rectangle
# from matplotlib.patches import Circle
# from PIL import Image
# from numpy import savez_compressed
# from numpy import asarray, ndarray, expand_dims
# from os import listdir
# #from mtcnn.mtcnn import MTCNN
# from config import AppConfig
#
#
# from matplotlib import pyplot as plt
# from numpy import asarray
# from numpy import expand_dims
# from numpy import reshape
# from numpy import load
#
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import Normalizer
# from sklearn.svm import SVC
#
# config = AppConfig(path='settings.ini', section='IMAGE_ENCODING')
# model = load_model('model/facenet_keras.h5')
# eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
import math
import time

import face_alignment
import imutils
from PIL import Image
from deepface.basemodels import Facenet
from deepface.commons import distance
from imutils import paths
import os
import numpy as np
from numpy import ndarray
import cv2
from matplotlib import pyplot as plt
#from keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import load_model


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn import metrics

from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Flatten

from config import AppConfig
from mask_detection import FaceDetector
from mtcnn import MTCNN
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split, GridSearchCV

from deepface import DeepFace
from deepface.commons.functions import detect_face, preprocess_face
import face_recognition as fr
import face_alignment


#label = decode_predictions(yhat)
current_dir = os.path.dirname(os.path.abspath(__file__))
config = AppConfig(path='settings.ini', section='IMAGE_ENCODING')


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(xx, yy)
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)

    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation


    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance



class FaceRecognition:

    #model = load_model(f'{current_dir}/models/facenet_keras.h5')

    # encs, labels = self.dataset['arr_0'], self.dataset['arr_1']
    #
    # self.data = {}
    # for label, enc in zip(labels, encs):
    #     if self.data.get(label) is None:
    #         self.data[label] = [enc]
    #     else:
    #         self.data[label].append(enc)


    def __init__(self):
        pass

        # models = ['VGG-Face', 'OpenFace','Facenet': Facenet.loadModel,
        #     'Facenet512': Facenet512.loadModel,
        #     'DeepFace': FbDeepFace.loadModel,
        #     'DeepID': DeepID.loadModel,
        #     'Dlib': DlibWrapper.loadModel,
        #     'ArcFace': ArcFace.loadModel,
        #     'Emotion': Emotion.loadModel,
        #     'Age': Age.loadModel,
        #     'Gender': Gender.loadModel,
        #     'Race': Race.loadModel
        # }

        self.deepface_model = DeepFace.build_model('Facenet')
        self.dataset = np.load(f"{current_dir}/{config['embeddings_dataset']}")

        embeddings_dataset = self.dataset
        embeddings_x, embeddings_y = embeddings_dataset['arr_0'], embeddings_dataset['arr_1']




        train_x, test_x, train_y, test_y = train_test_split(embeddings_x, embeddings_y,
                                                        test_size=0.25, random_state=42)  # , stratify=embeddings_y)

        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        train_x = in_encoder.transform(train_x)
        test_x = in_encoder.transform(test_x)

        # label encode targets
        self.out_encoder = LabelEncoder()
        self.out_encoder.fit(train_y)
        train_y = self.out_encoder.transform(train_y)
        test_y = self.out_encoder.transform(test_y)

        # fit model
        self.svc_model = SVC(kernel='linear', probability=True)
        self.svc_model.fit(train_x, train_y)

        # predict
        yhat_train = self.svc_model.predict(train_x)
        yhat_test = self.svc_model.predict(test_x)
        # score
        score_train = metrics.accuracy_score(train_y, yhat_train)
        score_test = metrics.accuracy_score(test_y, yhat_test)

        # # # safsa =23632


    @staticmethod
    def load_image(image_path: str):
        image = None

        try:
            image = cv2.imread(image_path)  # open the image
            image = image.astype(np.uint8)
            # image = imutils.resize(image, width=600)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert the image to RGB format
        except (AttributeError, cv2.error) as e:
            print(f'load image error: {e}')

        return image

    @staticmethod
    def preprocess_face(face_array: ndarray, target_size: tuple = (224, 224)):

        face_array = cv2.resize(face_array, dsize=target_size,
                                   interpolation=cv2.INTER_CUBIC)

        # scale pixel values
        face_array = face_array.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_array.mean(), face_array.std()
        face_array = (face_array - mean) / std

        #face_array = face_array / 255  # normalize input in [0, 1]
        face_array = np.expand_dims(face_array, axis=0)

        return face_array

    @staticmethod
    def detect_faces(image_array: ndarray):
        detector = MTCNN()
        faces = detector.detect_faces(image_array)  # get list of face boxes
        image_height, image_width = image_array.shape[:2]

        face_arrays = []
        for face in faces:
            x, y, w, h = face.get('box')

            start_x, start_y = (max(0, x), max(0, y))
            end_x, end_y = (min(image_width - 1, abs(start_x + w)), min(image_height - 1, abs(start_y + h)))
            extracted_face = image_array[start_y:end_y, start_x:end_x]

            face_arrays.append({
                'confidence': face.get('confidence'),
                'keypoints': face.get('keypoints'),
                'box': (x, y, w, h),
                'rect': (start_x, start_y, end_x, end_y),
                'rect_size': end_x - start_x + end_y - start_y,
                'array': extracted_face
            })

        return face_arrays

    @staticmethod
    def align_face(face_array: ndarray, face_keypoints: dict=None):
        #detector = MTCNN()
        # image_array = face_array.copy()
        #
        # t1 = time.time()
        # faces = detector.detect_faces(face_array)
        # box = faces[0]['box']
        # box = box[0], box[1], box[0] + box[2], box[1] + box[3]
        #
        # face_array = face_array[box[1]: box[3], box[0]: box[2]]
        #
        #
        # print(f'time1 = {time.time()-t1}')
        # face_keypoints = faces[0]['keypoints']

        left_eye_center = np.array(face_keypoints['left_eye'])
        right_eye_center = np.array(face_keypoints['right_eye'])


        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = np.arctan2(dy, dx) * 180. / np.pi

        # h, w = image_array.shape[:2]
        # rotate_center = (w // 2, h // 2)

        #eye_center = int((left_eye_center[0] + right_eye_center[0]) * 1. / 2), int((left_eye_center[1] + right_eye_center[1]) * 1. / 2)
        # rotate_matrix = cv2.getRotationMatrix2D(rotate_center, angle, scale=1)
        # face_aligned2 = cv2.warpAffine(image_array, rotate_matrix, (w, h))

        img = Image.fromarray(face_array)
        face_aligned = np.array(img.rotate(angle))

        return face_aligned


    @classmethod
    def crop_mask(cls, frame: np.ndarray, box: tuple, ratio: float=1.85) -> np.ndarray:
        """
        Rebuild detection box to square shape
        Args:
            frame: rgb image in np.uint8 format
            box: list with follow structure: [x1, y1, x2, y2]
        Returns:
            Image crop by box with square shape
        """
        x, y, w, h = box
        start_x, end_x = x, x + w
        start_y, end_y = y, y + h

        if ratio == 0:
            return frame[start_y:end_y, start_x:end_x]


        new_y = end_y - int((end_y - start_y) / ratio)
        cut_img = frame[start_y:new_y, start_x:end_x]

        return cut_img


    def get_embeddings(self, face_pixels: ndarray):
        embeddings = self.deepface_model.predict(face_pixels)

        return embeddings[0]


    def identify_face(self, face_array: ndarray, probability_level: float = 0.6):
        # face_array = cv2.resize(face_array, dsize=(160, 160), interpolation=cv2.INTER_CUBIC)
        # testx = np.asarray(face_array)
        # testx = np.expand_dims(testx, axis=0)
        # print("Input test data shape: ", testx.shape)

        # self.preprocess_face(face_array)

        # find embeddings
        sample_x = []
        #for test_pixels in testx:
        embeddings = self.get_embeddings(face_array)
        sample_x.append(embeddings)
        sample_x = np.asarray(sample_x)
        print("Input test embedding shape: ", sample_x.shape)


        yhat_class = self.svc_model.predict(sample_x)
        yhat_prob = self.svc_model.predict_proba(sample_x)

        class_index = yhat_class[0]
        probability = np.max(yhat_prob)
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = self.out_encoder.inverse_transform(yhat_class)

        person_name = predict_names[0] if probability >= probability_level else "Unknown"
        return {'name': person_name, 'probability': probability, 'embeddings': embeddings}



if __name__ == '__main__':
    recognizer = FaceRecognition()

    #$ --- NEW
    #detector = FaceDetector()

    image_array = recognizer.load_image('face_align.jpg')
    recognizer.align_face(image_array)

    # image_array = imutils.resize(image_array, width=600)
    # # boxes = fr.face_locations(image_array)
    # # encodings = fr.face_encodings(image_array, boxes)
    # # encodings2 = recognizer.get_embeddings_new(image_array, boxes)
    #
    # # plt.imshow(image_array)
    # # plt.show()
    # faces = recognizer.detect_faces(image_array)
    # #faces = detector.extract_faces(image_array)
    # #
    # x, y, w, h = faces[0]['rect']
    # box = (x, y, x + w, y + h)
    # box = [box[1], box[2], box[3], box[0]]
    #
    # # face_array = image_array[y:y + h, x:x + w]
    #
    # #res = recognizer.identify_new(face_array, 0.4)
    #
    #
    #
    # #boxes = fr.face_locations(image_array)
    # res = recognizer.identify_new2(image_array, [box], 0.4)

    fasf= 325632
    fa =23523

    #DatasetCompressor.start()


