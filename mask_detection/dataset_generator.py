import math
import random

import imutils
import numpy as np
import os
import cv2
import json
import face_alignment

from skimage import io
from numpy import ndarray
from cv2 import imread, imwrite
from matplotlib import pyplot as plt
import warnings

from mask_detection.mask_overlay import FaceOverlay

warnings.simplefilter("ignore")


import tensorflow as tf
from tensorflow import keras


class Utils:

    @staticmethod
    def save_json(path: str, save_object: dict or list) -> None:
        """
        Save json.

        :param path: local path with filename like 'dir/filename'
        :param save_object: object for save
        :return: None
        """

        with open(f"{path}.json", 'w', encoding='utf-8') as outfile:
            json.dump(save_object, outfile, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(path: str):
        """
        Load json from path.

        :param path: local path with filename
        :return: Any
        """
        with open(f"{path}.json", encoding='utf-8') as file:
            return json.load(file)

    @staticmethod
    def shape_to_list(shape):
        coords = []
        for i in range(0, 68):
            coords.append((shape.part(i).x, shape.part(i).y))

        return coords


class DatasetGenerator:

    @staticmethod
    def start(folder_path='dataset/mask_off/'):

        # loop on all files of the folder and build a list of files paths
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)\
                    if os.path.isfile(os.path.join(folder_path, f))]

        mask_types = ['mask_on', 'mask_mouth', 'mask_chin']
        mask_colors = ['black', 'blue', 'white']

        face_overlay = FaceOverlay()
        part = math.ceil(len(image_paths) / len(mask_colors))
        random.shuffle(image_paths)
        for index, image_path in enumerate(image_paths):
            print(f"[INFO] processing image [{index} / {len(image_paths)}]")

            for mask_type in mask_types:
                current_mask_color = mask_colors[math.floor(index / part)]
                mask_path = f"mask/{current_mask_color}_{mask_type}.png"
                save_path = f"dataset/{mask_type}"

                face_overlay.put_mask(image_path, mask_path, save_path)



if __name__ == '__main__':
    DatasetGenerator.start()
