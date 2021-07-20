import imutils
from cv2 import imread, imwrite
from matplotlib import pyplot as plt

import skimage as sk
from skimage import transform
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage
import random
from skimage import img_as_ubyte
import os

from numpy import ndarray
from scipy import ndimage
import random
import os
import cv2

"""
     rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
            
"""
MAX_SIZE = (600, 600)

class AugmentationGenerator:

    @staticmethod
    def start(image_number: int = 50, folder_path='examples'):

        # loop on all files of the folder and build a list of files paths
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                  os.path.isfile(os.path.join(folder_path, f)) and 'augmented_image' not in os.path.join(folder_path, f)]

        for num_generated_file in range(image_number):
            # random image from the folder
            image_path = random.choice(image_paths)
            # read image as an two dimensional array of pixels
            image_to_transform = imread(image_path) / 255.0
            image_to_transform = imutils.resize(image_to_transform, width=600, inter=cv2.INTER_CUBIC)

            # available_transformations = {
            #     'rotate': AugmentationGenerator.add_random_rotation,
            #     'horizontal_flip': AugmentationGenerator.add_horizontal_flip,
            #     'noise_or_blur': random.choice([AugmentationGenerator.add_random_noise,
            #                                               AugmentationGenerator.add_blur])
            # }
            available_transformations = {
                'rotate': AugmentationGenerator.add_random_rotation,
                'horizontal_flip': AugmentationGenerator.add_horizontal_flip,
                'noise': AugmentationGenerator.add_random_noise,
                'shear': AugmentationGenerator.add_shear,
                'horizontal_shift': AugmentationGenerator.add_horizontal_shift,
                'vertical_shift': AugmentationGenerator.add_vertical_shift,
                'zoom': AugmentationGenerator.add_scale,
                'blur': AugmentationGenerator.add_blur
            }


            # random num of transformations to apply
            num_transformations_to_apply = random.randint(2, len(available_transformations) + 1)

            num_transformations = 0
            transformed_image = None

            while available_transformations and num_transformations <= num_transformations_to_apply:
                # choose a random transformation to apply for a single image

                key = random.choice(list(available_transformations))
                transformed_image = available_transformations[key](image_to_transform)
                num_transformations += 1

                del available_transformations[key]

            file_name = f"augmented_image_{image_path.split(os.path.sep)[-1].split('.')[0]}_{num_generated_file + 1}.jpg"
            AugmentationGenerator.save_image(transformed_image * 255.0, folder_path, file_name)


    @staticmethod
    def save_image(transformed_image: ndarray, folder_path: str, file_name: str):

        if transformed_image is not None:
            print(f"[INFO] Generated augmented image: {file_name}")
            new_file_path = f'{folder_path}/{file_name}'

            imwrite(new_file_path, transformed_image)
        else:
            print('image is not defined!')

    @staticmethod
    def add_shear(image_array: ndarray):
        # image shearing using sklearn.transform.AffineTransform
        # try out with differnt values of shear
        shear_level = random.uniform(-15, 15) / 100
        tf = AffineTransform(shear=shear_level)
        return transform.warp(image_array, tf, order=1, preserve_range=True, mode='wrap')

    @staticmethod
    def add_scale(image_array: ndarray):
        # Image rescaling with sklearn.transform.rescale
        scale = 1.0 + random.uniform(-15, 15) / 100

        return transform.rescale(image_array, scale)

    @staticmethod
    def _add_shift(image_array: ndarray, vector):
        transform = AffineTransform(translation=vector)
        shifted = warp(image_array, transform, mode='wrap', preserve_range=True)

        return shifted.astype(image_array.dtype)

    @staticmethod
    def add_horizontal_shift(image_array: ndarray, offset: float=0.2):
        vector = (int(image_array.shape[1] / offset), 0)
        return AugmentationGenerator._add_shift(image_array, vector)

    @staticmethod
    def add_vertical_shift(image_array: ndarray, offset: float=0.2):
        vector = (0, int(image_array.shape[0] / offset))
        return AugmentationGenerator._add_shift(image_array, vector)


    @staticmethod
    def add_random_rotation(image_array: ndarray):
        # pick a random degree of rotation between 25% on the left and 25% on the right
        random_degree = random.uniform(-20, 20)
        return sk.transform.rotate(image_array, random_degree)

    @staticmethod
    def add_random_noise(image_array: ndarray):
        # add random noise to the image

        return sk.util.random_noise(image_array)

    @staticmethod
    def add_horizontal_flip(image_array: ndarray):
        return image_array[:, ::-1]

    @staticmethod
    def add_blur(image_array: ndarray, is_random: bool=True):
        sigma = 1.0
        random_seed = 0.0
        if is_random:
            random_seed = random.random()

        return sk.filters.gaussian(
            image_array, sigma=sigma + random_seed, truncate=3.5, multichannel=True)

    @staticmethod
    def show_image(image_array: ndarray, image_title: str='Image'):
        plt.imshow(image_array)
        plt.title(image_title)
        plt.show()


if __name__ == '__main__':
    AugmentationGenerator.start()