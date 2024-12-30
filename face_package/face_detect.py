import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from loguru import logger
from typing import List
from ultralytics.engine.model import Results

from face_package import DATA_DIR, MODEL_DIR


class Detect:
    def __init__(self):

        model_path = f"{MODEL_DIR}/yolov11m-face.pt"
        logger.info(f"model path is f{model_path}")
        self.yolo = YOLO(model_path)

        if torch.cuda.is_available():
            self.yolo.cuda()
            logger.info("Loaded model, cuda is available")
        self.yolo.eval()

    def detect(self, group_person_photo: Image, conf=0.2, iou=0.75) -> np.ndarray:
        """Get the specific subcoordinates of face and return the content as [x, y, width, height]

        Args:
            group_person_photo (str): Address of the photo file

        Returns:
            np.ndarray: Return the position of each face, which are [x, y, width, height].
        """
        result = self.detect_res(group_person_photo, conf, iou)
        logger.info(result)
        return result.boxes.xywh.cpu().numpy()

    def detect_res(self, group_person_photo: Image, conf=0.2, iou=0.75):

        result: List[Results] = self.yolo.predict(
            group_person_photo, conf=conf, iou = iou, verbose=False)
        logger.info(result)
        return result[0]
