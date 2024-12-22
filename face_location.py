import os
import cv2
import numpy as np
import torch
import face_recognition
from ultralytics import YOLO
from loguru import logger
from typing import List
from ultralytics.engine.model import Results


class Location:
    def __init__(self, model_path: str):
        # self.model_path = model_path
        # self.person_photo_path = person_photo_path
        self.yolo = YOLO(model_path)
        # self.yolo.load(self.model_path)
        if torch.cuda.is_available():
            self.yolo.cuda()
            logger.info("Loaded model, cuda is available")
            self.yolo.eval()
        # self.person_encoding_list = []
    #     self._load_person_photo()

    # def _load_person_photo(self):
    #     for file in os.listdir(self.person_photo_path):
    #         person_head = cv2.imread(
    #             os.path.join(self.person_photo_path, file))
    #         face_decoder = face_recognition.face_encodings(person_head)[0]
    #         name = file.removesuffix(".png").removesuffix(".jpg")
    #         self.person_encoding_list.append((face_decoder, name))

    def detect(self, group_person_photo: str) -> np.ndarray:
        """Get the specific subcoordinates of face and return the content as [x, y, width, height]

        Args:
            group_person_photo (str): Address of the photo file

        Returns:
            np.ndarray: Return the position of each face, which are [x, y, width, height].
        """
        result: List[Results] = self.yolo(group_person_photo)
        logger.info(result)
        return result[0].boxes.xywh.cpu().numpy()

