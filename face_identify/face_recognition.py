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

from imutils import paths
import os
import numpy as np
from numpy import ndarray
import cv2
from matplotlib import pyplot as plt
#from keras.models import load_model
from tensorflow.keras.models import load_model


from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

from config import AppConfig
from mask_detection import FaceDetector

config = AppConfig(path='settings.ini', section='IMAGE_ENCODING')
current_dir = os.path.dirname(os.path.abspath(__file__))


class FaceRecognition:

    model = load_model(f'{current_dir}/models/facenet_keras.h5')

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
    def detect_faces(image_array: ndarray):
        detector = MTCNN()
        faces = detector.detect_faces(image_array) # get list of face boxes

        face_arrays = []
        for face in faces:
            x, y, w, h = face.get('box')
            start_x, start_y = abs(x), abs(y)
            end_x, end_y = abs(start_x + w), abs(start_y + h)

            extracted_face = image_array[start_y:end_y, start_x:end_x]
            resized_image = cv2.resize(extracted_face, dsize=(160, 160),
                                        interpolation=cv2.INTER_CUBIC)
            face_arrays.append({
                'confidence': face.get('confidence'),
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

    @classmethod
    def identify(cls, face_array: ndarray):
        testx = np.asarray(face_array)
        testx = testx.reshape((-1, 160, 160, 3))
        print("Input test data shape: ", testx.shape)

        # find embeddings
        sample_x = []
        for test_pixels in testx:
            embeddings = FaceRecognition.extract_embeddings(test_pixels)
            sample_x.append(embeddings)
        sample_x = np.asarray(sample_x)
        print("Input test embedding shape: ", sample_x.shape)

        faces_dataset = np.load(config['faces_dataset'])
        embeddings_dataset = np.load(config['embeddings_dataset'])

        train_faces_x, train_faces_y = faces_dataset['arr_0'], faces_dataset['arr_1']
        train_emdeddings_x, train_embeddings_y = embeddings_dataset['arr_0'], embeddings_dataset['arr_1']
        print("Loaded data: Train=%d , Test=%d" % (train_emdeddings_x.shape[0], sample_x.shape[0]))

        # normalize the input data
        in_encode = Normalizer(norm='l2')
        train_emdeddings_x = in_encode.transform(train_emdeddings_x)
        sample_x = in_encode.transform(sample_x)

        # create a label vector
        new_testy = train_embeddings_y
        out_encode = LabelEncoder()
        out_encode.fit(train_embeddings_y)
        train_embeddings_y = out_encode.transform(train_embeddings_y)
        new_testy = out_encode.transform(new_testy)

        # define svm classifier model
        model = SVC(kernel='linear', probability=True)
        model.fit(train_emdeddings_x, train_embeddings_y)

        # predict
        predict_train = model.predict(train_emdeddings_x)
        predict_test = model.predict(sample_x)

        # get the confidence score
        probability = model.predict_proba(sample_x)
        confidence = np.max(probability)

        # Accuracy
        acc_train = accuracy_score(train_embeddings_y, predict_train)

        # display
        trainy_list = list(train_embeddings_y)
        p = int(predict_test)
        val = 0
        if p in trainy_list:
            val = trainy_list.index(p)

        # display Input Image
        plt.subplot(1, 2, 1)
        plt.imshow(face_array)
        predict_test = out_encode.inverse_transform(predict_test)
        plt.title(predict_test)
        plt.xlabel("Input Image")

        # display Predicated data
        plt.subplot(1, 2, 2)
        plt.imshow(train_faces_x[val])
        train_embeddings_y = out_encode.inverse_transform(train_embeddings_y)
        plt.title(train_embeddings_y[val])
        plt.xlabel("Predicted Data")

        plt.show()



class DatasetCompressor:

    @staticmethod
    def start():

        print('[INFO] Creating faces dataset')
        train_x, train_y = DatasetCompressor.create_faces_dataset(config['dataset_path'])

        print('[INFO] Creating embeddings dataset')
        new_train_x, new_train_y = DatasetCompressor.create_embeddings_dataset(train_x, train_y)

        DatasetCompressor.save_dataset(config['faces_dataset'], train_x, train_y)
        DatasetCompressor.save_dataset(config['embeddings_dataset'], new_train_x, new_train_y)

    @staticmethod
    def create_faces_dataset(dataset_path: str):
        image_paths = list(paths.list_images(dataset_path))

        X = []
        y = []
        for k, image_path in enumerate(image_paths):
            print(f"[INFO] processing image {k + 1}/{len(image_paths)}")

            label = image_path.split(os.path.sep)[-2]
            image_array = FaceRecognition.load_image(image_path)
            if image_array is not None:
                faces = FaceRecognition.detect_faces(image_array)
                if faces:
                    max_size_face = max(faces, key=lambda x: x['rect_size'])

                    # plt.imshow(max_size_face['array'])
                    # plt.title(image_path.split(os.path.sep)[-1])
                    # plt.show()

                    X.append(max_size_face['array'])
                    y.append(label)
                else:
                    print(f'[WARNING] Face is not found: {image_path}')

        return np.asarray(X), np.asarray(y)

    @staticmethod
    def save_dataset(file_name: str, train_X: ndarray, train_Y: ndarray):
        # compress data
        np.savez_compressed(file_name, train_X, train_Y)

    @staticmethod
    def create_embeddings_dataset(train_x: ndarray, train_y: ndarray):
        # convert each face in the train set to an embedding
        new_train_x = []
        for face_pixels in train_x:
            embedding = FaceRecognition.extract_embeddings(face_pixels)
            new_train_x.append(embedding)

        return np.asarray(new_train_x), np.asarray(train_y)




if __name__ == '__main__':
    recognizer = FaceRecognition()

    # image_array = recognizer.load_image('test3.png')
    # recognizer.align_face(image_array)

    # image_array = recognizer.load_image('example2.jpg')
    # faces = recognizer.detect_faces(image_array)
    # recognizer.extract_embeddings(faces[0]['array'])

    detector = FaceDetector()

    image_array = recognizer.load_image('example2.jpg')
    faces = detector.extract_faces(image_array)

    x, y, w, h = faces[0]['rect']
    face_array = image_array[y:y + h, x:x + w]
    face_array = cv2.resize(face_array, dsize=(160, 160), interpolation=cv2.INTER_CUBIC)

    recognizer.identify(face_array)

    #DatasetCompressor.start()


