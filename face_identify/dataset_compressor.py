import face_recognition
import imutils

from config import AppConfig
from imutils import paths
import os
import numpy as np
from numpy import ndarray

from face_identify import FaceRecognition

config = AppConfig(path='settings.ini', section='IMAGE_ENCODING')


class DatasetCompressor:

    @staticmethod
    def start(dataset_path: str = config['dataset_path'], person_name: str=None):

        print('[INFO] Creating embeddings dataset')
        old_x, old_y = DatasetCompressor.load_dataset(config['embeddings_dataset'])
        train_x, train_y = DatasetCompressor.create_embeddings_dataset(dataset_path, person_name)

        new_train_x = np.append(old_x, train_x, 0) if old_x is not None else train_x
        new_train_y = np.append(old_y, train_y, 0) if old_y is not None else train_y
        DatasetCompressor.save_dataset(config['embeddings_dataset'], new_train_x, new_train_y)

    @staticmethod
    def create_embeddings_dataset(dataset_path: str, default_label: str=None):
        image_paths = list(paths.list_images(dataset_path))

        recognition = FaceRecognition()

        X = []
        y = []
        for k, image_path in enumerate(image_paths):
            print(f"[INFO] processing image {image_path} --- {k + 1}/{len(image_paths)}")

            label = image_path.split(os.path.sep)[-2] if not default_label else default_label
            image_array = recognition.load_image(image_path)
            if image_array is not None:

                faces = recognition.detect_faces(image_array)
                if faces:
                    max_size_face = max(faces, key=lambda x: x['rect_size'])

                    # plt.imshow(max_size_face['array'])
                    # plt.title(image_path.split(os.path.sep)[-1])
                    # plt.show()

                    #new_box = tuple(list(max_size_face['rect'][1:]) + [max_size_face['rect'][0]])
                    face_array = max_size_face['array']
                    face_keypoints = max_size_face['keypoints']
                    #face_array = recognition.align_face(face_array, face_keypoints)
                    face_array = recognition.preprocess_face(face_array, target_size=(224, 224))
                    embedding = recognition.get_embeddings(face_array)
                    X.append(embedding)
                    y.append(label)
                else:
                    print(f'[WARNING] Face is not found: {image_path}')

        return np.asarray(X), np.asarray(y)

    @staticmethod
    def save_dataset(file_name: str, train_X: ndarray, train_Y: ndarray):
        # compress data
        np.savez_compressed(file_name, train_X, train_Y)

    @staticmethod
    def load_dataset(file_path: str):
        # load data
        result = None, None
        try:
            dataset = np.load(file_path)
            X, y = dataset['arr_0'], dataset['arr_1']
            result = np.asarray(X), np.asarray(y)
        except Exception as e:
            print(f'[ERROR] dataset loading: {e}')

        return result



if __name__ == '__main__':
    DatasetCompressor.start() # "test", person_name="Volodymyr_Shevchenko"
