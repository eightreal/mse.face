import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from loguru import logger
from typing import List
from ultralytics.engine.model import Results


class Location:
    def __init__(self):
        # self.model_path = model_path
        # self.person_photo_path = person_photo_path
        model_path = "model/yolov11n-face.pt"
        self.yolo = YOLO(model_path)
        # self.yolo.load(self.model_path)
        if torch.cuda.is_available():
            self.yolo.cuda()
            logger.info("Loaded model, cuda is available")
        self.yolo.eval()

    def detect(self, group_person_photo: Image) -> np.ndarray:
        """Get the specific subcoordinates of face and return the content as [x, y, width, height]

        Args:
            group_person_photo (str): Address of the photo file

        Returns:
            np.ndarray: Return the position of each face, which are [x, y, width, height].
        """
        result: List[Results] = self.yolo(group_person_photo)
        logger.info(result)
        return result[0].boxes.xywh.cpu().numpy()

