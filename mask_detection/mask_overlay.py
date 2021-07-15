import os
import sys
import argparse
import numpy as np
import cv2
import face_alignment

import math
from PIL import Image, ImageFile


def align_face(face, landmarks):
    left_eye_center2 = (0.5 * (landmarks[36][0] + landmarks[39][0]), 0.5 * (landmarks[36][1] + landmarks[39][1]))
    right_eye_center2 = (0.5 * (landmarks[42][0] + landmarks[45][0]), 0.5 * (landmarks[42][1] + landmarks[45][1]))

    left_eye_center = 1./ 2 * (landmarks[36] + landmarks[39])
    right_eye_center = 1./ 2 * (landmarks[42] + landmarks[45])

    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    angle = math.atan2(dy, dx) * 180. / math.pi


    eye_center = ((landmarks[36][0] + landmarks[45][0]) * 1. / 2, (landmarks[36][1] + landmarks[45][1]) * 1. / 2)
    # dx = landmarks[45][0] - landmarks[36][0]
    # dy = landmarks[45][1] - landmarks[36][1]
    # angle = math.atan2(dy, dx) * 180. / math.pi
    RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    face_aligned = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))

    return face_aligned


class FaceOverlay:

    def __init__(self):
        self.face_img: ImageFile = None
        self.mask_img: ImageFile = None
        self.face_align = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

    def put_mask(self, image_path: str, mask_path: str, save_path: str):
        self.face_img = Image.open(image_path)
        self.mask_img = Image.open(mask_path)

        image_array = np.array(self.face_img)
        landmarks = self.face_align.get_landmarks(image_array)

        image_path = image_path.replace('/', os.path.sep)
        save_path = save_path.replace('/', os.path.sep)
        file_name = image_path.split(os.path.sep)[-1]

        self._add_mask(landmarks[0])
        self._save_image(save_path, file_name)

        return self.face_img

    def _save_image(self, save_path: str, file_name: str):
        new_face_path = os.path.sep.join([save_path, file_name])
        self.face_img.save(new_face_path)
        print(f'Saved to {new_face_path}')

    def _add_mask(self, face_landmarks: list):
        nose_point = np.array(face_landmarks[28])
        chin_bottom_point = np.array(face_landmarks[8])
        chin_left_point = face_landmarks[2]
        chin_right_point = face_landmarks[14]

        # split mask and resize
        width, height = self.mask_img.size
        width_ratio = 1.1
        height_ratio = 1.06
        # Euclidean Distance
        new_height = int(height_ratio * np.linalg.norm(nose_point - chin_bottom_point))

        # left
        mask_left_img = self.mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self.mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        left_eye_center = 1. / 2 * (face_landmarks[36] + face_landmarks[39])
        right_eye_center = 1. / 2 * (face_landmarks[42] + face_landmarks[45])

        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = -1.0 * math.atan2(dy, dx) * 180. / math.pi
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        #ret = align_face(np.array(self.face_img), face_landmarks)

        # calculate mask location
        # lips_point = 0.5 * (np.array(face_landmarks[57]) + np.array(face_landmarks[8]))
        # center_x, center_y = lips_point[0], lips_point[1]

        lips_point = 0.5 * (np.array(face_landmarks[62]) + np.array(face_landmarks[66]))
        center_x, center_y = lips_point[0], lips_point[1]

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = int(center_x) + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = int(center_y) + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self.face_img.paste(rotated_mask_img, (box_x, box_y), rotated_mask_img)

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


if __name__ == '__main__':
    image_path = "dataset/mask_off/real_00849.jpg" # 00857 # 00849 #00793 # 00715 # 00657
    #mask_path = "mask/black_mask_on_new.png"
    mask_path = "mask/black_mask_on.png"

    face_overlay = FaceOverlay()
    face_overlay.put_mask(image_path, mask_path, "test")