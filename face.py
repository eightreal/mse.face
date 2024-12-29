"""Process processing class
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List
from loguru import logger
from face_location import Location
from face_emotion import Emotion
from face_recognition import Recognition
from face_matting import Matting


class Face:
    def __init__(self, location_model, emotion_model, recognition_model, matting_model):
        self.location_model: Location = location_model
        self.emotion_model: Emotion = emotion_model
        self.recognition_model: Recognition = recognition_model
        self.matting_model: Matting = matting_model
        self.origin_image: Image.Image
        self.name_mapping: Dict[str, np.ndarray] = {}
        self.group_person_xywhs: np.ndarray
        self.person_img_embs: Dict = {}
        self.info: Dict = {}

    def add_group_img(self, group_image: Image.Image):
        self.origin_image = group_image
        self.group_person_xywhs = self.location_model.detect(group_image)
        logger.info(f"detect {len(self.group_person_xywhs)} persons")

    def add_person_img(self, person_images: dict[str, Image.Image]):
        for item in person_images:
            name = item
            logger.info(f"decoder for {item}")
            image = person_images[item]
            person_xywh = self.location_model.detect(image)[0]
            box = (person_xywh[0], person_xywh[1], person_xywh[0] +
                   person_xywh[2], person_xywh[1] + person_xywh[3])
            image = image.crop(box)
            self.person_img_embs[name] = {
                "tensor": self.recognition_model.decode(image)}

    def detect(self):
        xywhs = self.group_person_xywhs.copy()
        person_encoder = self.person_img_embs.copy()
        ratio = 0.8
        while len(xywhs) > 0 and len(person_encoder) > 0 and ratio > 0.002:
            now_xywhs = xywhs.copy()
            unknow_xywhs = []

            for item in now_xywhs:
                person_xywh = item
                sub_image = self.get_expanded_sub_image(
                    person_xywh[0], person_xywh[1], person_xywh[2], person_xywh[3], 1.0)
                sub_image_tensor = self.recognition_model.decode(sub_image)
                name = "unknow"
                best_score = 0.0

                for item in person_encoder:
                    person_image_embs = person_encoder[item]["tensor"]
                    score = F.cosine_similarity(
                        person_image_embs, sub_image_tensor)
                    if score >= ratio and score > best_score:
                        name = item
                        best_score = score
                if name != "unknow":
                    self.info[name] = {"score": best_score,
                                       "xywh": person_xywh}
                    del person_encoder[name]
                    logger.info(f"name is {name}")
                else:
                    unknow_xywhs.append(person_xywh)

                ratio = ratio * 0.8

            xywhs = unknow_xywhs.copy()
        print(self.info)
        logger.info(f"get name for {len(self.info)} person")
        # print(unknow_xywhs)

    def get_expanded_sub_image(self, x, y, w, h, expansion_factor=1.2):
        # 计算扩充后的宽度和高度
        new_w = int(w * expansion_factor)
        new_h = int(h * expansion_factor)
        # 计算原始区域中心
        center_x = x + w // 2
        center_y = y + h // 2
        # 计算新的左上角坐标
        new_x = center_x - new_w // 2
        new_y = center_y - new_h // 2
        # 确保新的坐标在图像范围内
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_x = min(new_x, self.origin_image.width - new_w)
        new_y = min(new_y, self.origin_image.height - new_h)
        box = (new_x, new_y, new_x + new_w, new_y + new_h)
        sub_image = self.origin_image.crop(box)
        return sub_image
