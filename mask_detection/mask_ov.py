import os
import sys
import argparse
import numpy as np
import cv2
import face_alignment

import math
from PIL import Image, ImageFile



def rect_to_bbox(rect):
    """获得人脸矩形的坐标信息"""
    # print(rect)
    x = rect[3]
    y = rect[0]
    w = rect[1] - x
    h = rect[2] - y
    return (x, y, w, h)


# def align_face(faces):
#     predictor = dlib.shape_predictor("/Users/wuhao/lab/wear-a-mask/spider/shape_predictor_68_face_landmarks.dat")
#     faces_aligned = []
#     for face in faces:
#         rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
#         shape = predictor(np.uint8(face), rec)
#         order = [36, 45, 30, 48, 54]
#         for j in order:
#             x = shape.part(j).x
#             y = shape.part(j).y
#         eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2, (shape.part(36).y + shape.part(45).y) * 1. / 2)
#         dx = (shape.part(45).x - shape.part(36).x)
#         dy = (shape.part(45).y - shape.part(36).y)
#         angle = math.atan2(dy, dx) * 180. / math.pi
#         RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
#         RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))
#         faces_aligned.append(RotImg)
#     return faces_aligned

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

        self._add_mask3(landmarks[0])
        #self.face_img.show()
        #self._save_image(save_path, file_name)

        return self.face_img

    def _save_image(self, save_path: str, file_name: str):
        new_face_path = os.path.sep.join([save_path, file_name])
        self.face_img.save(new_face_path)
        print(f'Saved to {new_face_path}')

    def _add_mask3(self, face_landmarks: list):
        nose_point = np.array(face_landmarks[57])  # 0.5 * (np.array(face_landmarks[33]) + np.array(face_landmarks[51]))
        chin_bottom_point = np.array(face_landmarks[8])
        chin_left_point = face_landmarks[3]
        chin_right_point = face_landmarks[13]

        # split mask and resize
        width, height = self.mask_img.size
        width_ratio = 0.85
        height_ratio = 1.0
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

        ret = align_face(np.array(self.face_img), face_landmarks)

        # calculate mask location
        lips_point2 = 0.5 * (np.array(face_landmarks[5]) + np.array(face_landmarks[11]))
        lips_point = 0.5 * (np.array(face_landmarks[57]) + np.array(face_landmarks[8]))
        lips_point3 = 0.5 * (np.array(lips_point) + np.array(lips_point2))

        lips_point3[0] = face_landmarks[8][0]
        lips_point3[1] = (0.5 * (np.array(face_landmarks[6]) + np.array(face_landmarks[10])))[1]
        center_x, center_y = lips_point3[0], lips_point3[1]
        center_x2, center_y2 = lips_point2[0], lips_point2[1]
        center_x3, center_y3 = lips_point[0], lips_point[1]

        diff1 = np.linalg.norm(lips_point2 - face_landmarks[57])
        diff2 = np.linalg.norm(lips_point - face_landmarks[57])

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = int(center_x) + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = int(center_y) + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        copy_img = self.face_img.copy()
        # add mask
        self.face_img.paste(rotated_mask_img, (box_x, box_y), rotated_mask_img)

        fc_array = np.array(copy_img)
        for k, point in enumerate(face_landmarks):
            x, y = (int(point[0]), int(point[1]))
            fc_array = cv2.circle(fc_array, (x, y), 1, (0, 0, 255), 2)
            cv2.putText(fc_array, f"{k}", (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        fc_array = cv2.circle(fc_array, (int(center_x), int(center_y)), 1, (0, 255, 0), 2)
        fc_array = cv2.circle(fc_array, (int(center_x2), int(center_y2)), 1, (0, 255, 0), 4)
        fc_array = cv2.circle(fc_array, (int(center_x3), int(center_y3)), 1, (0, 0, 255), 3)

        cv2.imshow(f"new image66!", ret)
        cv2.imshow(f"new image55!", np.array(rotated_mask_img))
        cv2.imshow(f"new image22!", fc_array)
        cv2.imshow(f"new image!", np.array(self.face_img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _add_mask2(self, face_landmarks: list):
        nose_point = np.array(face_landmarks[51]) #0.5 * (np.array(face_landmarks[33]) + np.array(face_landmarks[51]))
        chin_bottom_point = np.array(face_landmarks[8])
        chin_left_point = face_landmarks[3]
        chin_right_point = face_landmarks[13]

        # split mask and resize
        width, height = self.mask_img.size
        width_ratio = 1.05
        height_ratio = 1.0
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

        ret = align_face(np.array(self.face_img), face_landmarks)

        # calculate mask location
        lips_point = 0.5 * (np.array(face_landmarks[57]) + np.array(face_landmarks[8]))
        center_x, center_y = lips_point[0], lips_point[1]

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = int(center_x) + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = int(center_y) + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        copy_img = self.face_img.copy()
        # add mask
        self.face_img.paste(rotated_mask_img, (box_x, box_y), rotated_mask_img)

        fc_array = np.array(copy_img)
        for k, point in enumerate(face_landmarks):
            x, y = (int(point[0]), int(point[1]))
            fc_array = cv2.circle(fc_array, (x, y), 1, (0, 0, 255), 2)
            cv2.putText(fc_array, f"{k}", (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        fc_array = cv2.circle(fc_array, (int(center_x), int(center_y)), 1, (0, 255, 0), 2)

        cv2.imshow(f"new image66!", ret)
        cv2.imshow(f"new image55!", np.array(rotated_mask_img))
        cv2.imshow(f"new image22!", fc_array)
        cv2.imshow(f"new image!", np.array(self.face_img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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

        ret = align_face(np.array(self.face_img), face_landmarks)

        # calculate mask location
        lips_point = 0.5 * (np.array(face_landmarks[57]) + np.array(face_landmarks[8]))
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
    image_path = "dataset/mask_off/real_00857.jpg" # 00857 # 00849 #00793 # 00715 # 00657
    #mask_path = "mask/black_mask_on_new.png"
    mask_path = "mask/black_mask_on.png"

    face_overlay = FaceOverlay()
    face_overlay.put_mask(image_path, mask_path, "")