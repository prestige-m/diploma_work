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


    def __init__(self):#, face_path, mask_path, show=False, model='hog',save_path = ''):
        # self.face_path = face_path
        # self.mask_path = mask_path
        # self.save_path = save_path
        # self.show = show
        # self.model = model
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None
        self.face_align = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

    def put_mask(self, image_path: str, mask_path: str):
        self.face_img = Image.open(image_path)
        self.mask_img = Image.open(mask_path)

        self._face_img = self.face_img.copy()
        self._mask_img = self.mask_img.copy()

        image_array = np.array(self.face_img)
        landmarks = self.face_align.get_landmarks(image_array)

        self._mask_face(landmarks[0])

        #
        # if found_face:
        #     # align
        #     src_faces = []
        #     src_face_num = 0
        #     with_mask_face = np.asarray(self._face_img)
        #     for (i, rect) in enumerate(face_locations):
        #         src_face_num = src_face_num + 1
        #         (x, y, w, h) = rect_to_bbox(rect)
        #         detect_face = with_mask_face[y:y + h, x:x + w]
        #         src_faces.append(detect_face)
        #     # 人脸对齐操作并保存
        #     faces_aligned = face_alignment(src_faces)
        #     face_num = 0
        #     for faces in faces_aligned:
        #         face_num = face_num + 1
        #         faces = cv2.cvtColor(faces, cv2.COLOR_RGBA2BGR)
        #         size = (int(128), int(128))
        #         faces_after_resize = cv2.resize(faces, size, interpolation=cv2.INTER_AREA)
        #         cv2.imwrite(self.save_path, faces_after_resize)
        #     # if self.show:
        #     #     self._face_img.show()
        #     # save
        #     # self._save()
        # else:
        #     #在这里记录没有裁的图片
        #     print('Found no face.'+self.save_path)

    def _mask_face(self, face_landmarks: list):
        nose_point = np.array(face_landmarks[28])
        chin_bottom_point = np.array(face_landmarks[8])
        chin_left_point = face_landmarks[2]
        chin_right_point = face_landmarks[14]

        # split mask and resize
        width, height = self._mask_img.size
        width_ratio = 1.1
        height_ratio = 1.1
        # Euclidean Distance
        new_height = int(height_ratio * np.linalg.norm(nose_point - chin_bottom_point))

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
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
        #angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0]) * 100 / np.pi
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        ret = align_face(np.array(self.face_img), face_landmarks)

        # calculate mask location
        #nose_point2 = np.array(face_landmarks[29])
        center_x2 = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y2 = (nose_point[1] + chin_bottom_point[1]) // 2

        lips_point = 0.5 * (np.array(face_landmarks[62]) + np.array(face_landmarks[66]))
        center_x, center_y = lips_point[0], lips_point[1]


        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = int(center_x) + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = int(center_y) + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(rotated_mask_img, (box_x, box_y), rotated_mask_img)

        fc_array = np.array(self.face_img)
        for k, center_coordinates in enumerate(face_landmarks):
            x, y = int(center_coordinates[0]), int(center_coordinates[1])
            #cv2.putText(original_image, f"#{k + 1}", (x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(fc_array, (x, y), 1, (0, 0, 255), 2)

        cv2.circle(fc_array, (box_x, box_y), 1, (0, 255, 0), 2)
        cv2.circle(fc_array, (int(center_x), int(center_y)), 1, (0, 255, 0), 2)
        cv2.circle(fc_array, (int(center_x2), int(center_y2)), 1, (255, 0, 0), 2)


        cv2.imshow(f"new image66!", ret)
        cv2.imshow(f"new image55!", np.array(rotated_mask_img))
        cv2.imshow(f"new image22!", fc_array)
        cv2.imshow(f"new image!", np.array(self._face_img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _save(self):
        path_splits = os.path.splitext(self.face_path)
        new_face_path = path_splits[0] + '-with-mask' + path_splits[1]
        self._face_img.save(new_face_path)
        print(f'Save to {new_face_path}')

    def get_image(self):
        return self._face_img

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
    image_path = "dataset/mask_off/real_00793.jpg" # 00857 # 00849 #00793
    #mask_path = "mask/black_mask_on_new.png"
    mask_path = "mask/black_mask_on.png"

    face_overlay = FaceOverlay()
    face_overlay.put_mask(image_path, mask_path)