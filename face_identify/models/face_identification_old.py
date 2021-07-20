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
import imutils
from deepface.basemodels import Facenet
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
from deepface.commons.functions import detect_face
import face_recognition as fr

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


    def __init__(self):
        pass

        # self.facenet_model = Facenet.loadModel()
        self.dataset = np.load(f"{current_dir}/{config['embeddings_dataset']}")

        embeddings_dataset = self.dataset
        embeddings_x, embeddings_y = embeddings_dataset['arr_0'], embeddings_dataset['arr_1']

        encs, labels = self.dataset['arr_0'], self.dataset['arr_1']

        self.data = {}
        for label, enc in zip(labels, encs):
            if self.data.get(label) is None:
                self.data[label] = [enc]
            else:
                self.data[label].append(enc)



        # train_x, test_x, train_y, test_y = train_test_split(embeddings_x, embeddings_y,
        #                                                 test_size=0.25, random_state=42)  # , stratify=embeddings_y)
        #
        # # normalize input vectors
        # in_encoder = Normalizer(norm='l2')
        # train_x = in_encoder.transform(train_x)
        # test_x = in_encoder.transform(test_x)
        #
        # # label encode targets
        # self.out_encoder = LabelEncoder()
        # self.out_encoder.fit(train_y)
        # train_y = self.out_encoder.transform(train_y)
        # test_y = self.out_encoder.transform(test_y)
        #
        # # fit model
        # self.svc_model = SVC(kernel='linear', probability=True)
        # self.svc_model.fit(train_x, train_y)
        #
        # # predict
        # yhat_train = self.svc_model.predict(train_x)
        # yhat_test = self.svc_model.predict(test_x)
        # # score
        # score_train = metrics.accuracy_score(train_y, yhat_train)
        # score_test = metrics.accuracy_score(test_y, yhat_test)
        #
        # safsa =23632


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
    def detect_faces(image_array: ndarray, image_size: tuple = (160, 160)):
        detector = MTCNN()
        faces = detector.detect_faces(image_array)  # get list of face boxes

        face_arrays = []
        for face in faces:
            x, y, w, h = face.get('box')
            start_x, start_y = abs(x), abs(y)
            end_x, end_y = abs(start_x + w), abs(start_y + h)

            extracted_face = image_array[start_y:end_y, start_x:end_x]
            resized_image = cv2.resize(extracted_face, dsize=image_size,
                                       interpolation=cv2.INTER_CUBIC)
            face_arrays.append({
                'confidence': face.get('confidence'),
                'box': (x, y, w, h),
                'rect': (start_x, start_y, end_x, end_y),
                'rect_size': end_x - start_x + end_y - start_y,
                'array': resized_image
            })

        return face_arrays

    @staticmethod
    def align_face(image_array: ndarray):
        detector = MTCNN()
        faces = detector.detect_faces(image_array)

        left_eye = faces[0]['keypoints']['left_eye']
        right_eye = faces[0]['keypoints']['right_eye']

        # cv2.circle(image_array, left_eye, 5, (255, 0, 0), -1)
        # cv2.circle(image_array, right_eye, 5, (255, 0, 0), -1)

        left_eye_x, left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye

        delta_x = right_eye_x - left_eye_x
        delta_y = right_eye_y - left_eye_y
        angle = np.arctan(delta_y / delta_x)
        angle = (angle * 180) / np.pi

        h, w = image_array.shape[:2]
        center = (w // 2, h // 2)
        # Defining a matrix M
        M = cv2.getRotationMatrix2D(center, (angle), 1.0)
        # Applying the rotation to our image using the cv2.warpAffine method
        rotated_image = cv2.warpAffine(image_array, M, (w, h))



        # scaling image
        faces = detector.detect_faces(rotated_image)

        new_left_eye = faces[0]['keypoints']['left_eye']
        new_right_eye = faces[0]['keypoints']['right_eye']

        delta_x_1 = new_right_eye[0] - new_left_eye[0]
        delta_y_1 = new_right_eye[1] - new_left_eye[1]

        dist_1 = np.sqrt((delta_x * delta_x) + (delta_y * delta_y))
        dist_2 = np.sqrt((delta_x_1 * delta_x_1) + (delta_y_1 * delta_y_1))
        ratio = dist_1 / dist_2

        # Defining aspect ratio of a resized image
        dim = (int(w * ratio), int(h * ratio))
        # We have obtained a new image that we call resized3
        resized = cv2.resize(rotated_image, dim)

        plt.imshow(resized)
        plt.title('Image')
        plt.show()


        fgadf =23462
        aqfasf=2365


    @classmethod
    def extract_embeddings(cls, face_pixels: ndarray):
        face_pixels = face_pixels.astype('float32')
        face_pixels = (face_pixels - face_pixels.mean()) / face_pixels.std()
        samples = np.expand_dims(face_pixels, axis=0)
        embedding = cls.model.predict(samples)

        return embedding[0]

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


        # rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        # landmarks = predictor(gray_frame, rect)
        # landmarks = face_utils.shape_to_np(landmarks)
        #
        # # 27 -  30  Nose bridge
        # for (x, y) in landmarks:
        #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        #
        # target_X, target_Y = landmarks[29]
        # # cut_img = frame[startY:target_Y, startX:endX]

        # ratio = (endY - startY) / (endY - target_Y)
        # print(f'ratio - {ratio}')

        new_y = end_y - int((end_y - start_y) / ratio)
        cut_img = frame[start_y:new_y, start_x:end_x]

        return cut_img #cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)


    def create_embeddings(self, face_pixels: ndarray):

        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std

        samples = np.expand_dims(face_pixels, axis=0)
        embeddings = self.facenet_model.predict(samples)

        return embeddings[0]


    def get_embeddings_new(self, image_array: ndarray, face_boxes: list):

        embeddings = fr.face_encodings(image_array, face_boxes)

        #samples = np.expand_dims(face_pixels, axis=0)
        #embeddings = self.facenet_model.predict(samples)

        return embeddings[0]

    def identify_new2(self, image_array: ndarray, boxes: list, probability_level: float = 0.6):
        #face_array = cv2.resize(face_array, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        embeddings = self.get_embeddings_new(image_array, boxes)

        result = {key: 0 for key in self.data}
        for label, enc in self.data.items():
            matches = fr.compare_faces(enc, embeddings, tolerance=0.6)
            result[label] = sum(matches) / len(matches)

        label_class = max(result, key=result.get)
        probability = result[label_class]

        person_name = label_class if probability > 0 else "Unknown"
        return {'name': person_name, 'probability': probability}


    def identify_new(self, face_array: ndarray, probability_level: float = 0.6):
        face_array = cv2.resize(face_array, dsize=(160, 160), interpolation=cv2.INTER_CUBIC)
        testx = np.asarray(face_array)
        testx = np.expand_dims(testx, axis=0)
        print("Input test data shape: ", testx.shape)

        # find embeddings
        sample_x = []
        for test_pixels in testx:
            embeddings = self.create_embeddings(test_pixels)
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
        return {'name': person_name, 'probability': probability}


    @classmethod
    def identify(cls, face_array: ndarray, confidence_level: float=0.6):
        face_array = cv2.resize(face_array, dsize=(160, 160), interpolation=cv2.INTER_CUBIC)
        testx = np.asarray(face_array)
        testx = np.expand_dims(testx, axis=0)
        #testx = testx.reshape((-1, 224, 224, 3))
        print("Input test data shape: ", testx.shape)

        # find embeddings
        sample_x = []
        for test_pixels in testx:
            #embeddings = FaceRecognition.extract_embeddings(test_pixels)
            embeddings = FaceRecognition.create_embeddings(test_pixels)
            sample_x.append(embeddings)
        sample_x = np.asarray(sample_x)
        print("Input test embedding shape: ", sample_x.shape)

        # embeddings_dataset = None
        # if cls.dataset is None:
        #     embeddings_dataset = np.load(config['embeddings_dataset'])
        #     cls.dataset = embeddings_dataset
        # else:
        embeddings_dataset = cls.dataset

        emdeddings_x, embeddings_y = embeddings_dataset['arr_0'], embeddings_dataset['arr_1']
        print("Loaded data: Train=%d , Test=%d" % (emdeddings_x.shape[0], sample_x.shape[0]))

        # t2 = {}
        # idx = 0
        # for it in embeddings_y:
        #     if it not in t2:
        #         t2[it] = idx
        #         idx += 1
        #
        # embeddings_y = [t2[it] for it in embeddings_y]


        # normalize the input data
        in_encode = Normalizer(norm='l2')
        emdeddings_x = in_encode.transform(emdeddings_x)
        sample_x = in_encode.transform(sample_x)

        new_testy = embeddings_y.copy()
        # create a label vector
        out_encode = LabelEncoder()

        out_encode.fit(embeddings_y)
        embeddings_y = out_encode.transform(embeddings_y)
        new_testy = out_encode.transform(new_testy)


        X_train, X_test, y_train, y_test = train_test_split(emdeddings_x, embeddings_y,
                                                            test_size=0.25, random_state=42)#, stratify=embeddings_y)
        # #
        # # sc = StandardScaler()
        # # sc.fit(X_train)
        # # X_train_std = sc.transform(X_train)
        # # X_test_std = sc.transform(X_test)

        # Instantiate the Support Vector Classifier (SVC)
        svc = SVC(probability=True, class_weight='balanced') # C=1.0, kernel='linear',
        #param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
        #              "kernel": ["rbf", "poly", "linear"]}

        ##
        param_grid = {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                      "gamma": [0.001, 0.01, 1, 10, 100], "kernel": ["rbf", "poly", "linear"]}
        model = GridSearchCV(svc, param_grid)

        # Fit the model
        clf = model.fit(X_train, y_train)

        temp = model.best_estimator_
        temp2 = model.best_params_

        # Make the predictions
        y_predict = model.predict(X_test)


        predict_test = model.predict(sample_x)
        # get the confidence score
        probability = model.predict_proba(sample_x)
        confidence = np.max(probability)


        acc = metrics.accuracy_score(y_test, y_predict)
        #acc2 = metrics.accuracy_score(embeddings_y, predict_test)


        # # normalize the input data
        # in_encode = Normalizer(norm='l2')
        # train_emdeddings_x = in_encode.transform(train_emdeddings_x)
        # sample_x = in_encode.transform(sample_x)
        #
        # # create a label vector
        # new_testy = train_embeddings_y
        # out_encode = LabelEncoder()
        # out_encode.fit(train_embeddings_y)
        # train_embeddings_y = out_encode.transform(train_embeddings_y)
        # new_testy = out_encode.transform(new_testy)
        #
        # # define svm classifier model
        # model = SVC(kernel='linear', probability=True)
        # model.fit(train_emdeddings_x, train_embeddings_y)
        #
        # # predict
        # predict_train = model.predict(train_emdeddings_x)
        # predict_test = model.predict(sample_x)
        #
        # # get the confidence score
        # probability = model.predict_proba(sample_x)
        # confidence = np.max(probability)
        #
        # # Accuracy
        # acc_train = metrics.accuracy_score(train_embeddings_y, predict_train)
        #
        # # Model Precision: what percentage of positive tuples are labeled as such?
        # print("Precision:", metrics.precision_score(y_test, y_pred))
        #
        # # Model Recall: what percentage of positive tuples are labelled as such?
        # print("Recall:", metrics.recall_score(y_test, y_pred))


        #-------------------------------
        # model = KNeighborsClassifier(n_neighbors=1,
        #                              n_jobs=-1)
        # model.fit(X_train, y_train)
        # acc = model.score(X_train, y_train)
        #
        # predict_test = model.predict(sample_x)
        #
        # # get the confidence score
        # probability = model.predict_proba(sample_x)
        # confidence = np.max(probability)
        #
        # y_predict = model.predict(X_test)
        # acc22 = metrics.accuracy_score(y_test, y_predict)

        person_name = out_encode.inverse_transform(predict_test)[0] if confidence >= confidence_level else "Unknown"
        return {'name': person_name, 'confidence':  confidence}



if __name__ == '__main__':
    recognizer = FaceRecognition()
    #recognizer.recognize()
    # image_array = recognizer.load_image('test3.png')
    # recognizer.align_face(image_array)

    #image_array = recognizer.load_image('test.jpg')
    # faces = recognizer.detect_faces(image_array, (224, 224))
    # res = recognizer.get_embeddings(faces[0]['array'])

    # faces2 = recognizer.detect_faces(image_array)
    # res = recognizer.create_embeddings(faces2[0]['array'])
    # res2 = recognizer.extract_embeddings(faces2[0]['array'])

    #$ --- NEW
    detector = FaceDetector()

    image_array = recognizer.load_image('example6.jpg')
    image_array = imutils.resize(image_array, width=600)
    # boxes = fr.face_locations(image_array)
    # encodings = fr.face_encodings(image_array, boxes)
    # encodings2 = recognizer.get_embeddings_new(image_array, boxes)

    # plt.imshow(image_array)
    # plt.show()
    faces = recognizer.detect_faces(image_array)
    #faces = detector.extract_faces(image_array)
    #
    x, y, w, h = faces[0]['rect']
    box = (x, y, x + w, y + h)
    box = [box[1], box[2], box[3], box[0]]

    # face_array = image_array[y:y + h, x:x + w]

    #res = recognizer.identify_new(face_array, 0.4)



    #boxes = fr.face_locations(image_array)
    res = recognizer.identify_new2(image_array, [box], 0.4)

    fasf= 325632
    fa =23523

    #DatasetCompressor.start()


