"""Process processing class
"""

import numpy as np

import torch.nn.functional as F
import torch
from PIL import Image,ImageDraw
from typing import Dict, List, Tuple
from loguru import logger


from face_package import crop_image_from_xywh, get_expanded_sub_image, xywh2str, str2xywh
from face_package.face_detect import Detect
from face_package.face_emotion import Emotion
from face_package.face_recognition import Recognition
from face_package.face_matting import Matting
from face_package.face_location import Location


# 创建一个集合


class Face:
    def __init__(self):

        self.detect_model: Detect = Detect()
        self.location_model: Location = Location()
        self.emotion_model: Emotion = Emotion()
        self.recognition_model: Recognition = Recognition()
        self.matting_model: Matting = Matting()
        self.origin_image: Image.Image = None
        self.name_mapping: Dict[str, np.ndarray] = {}
        self.group_person_xywhs: np.ndarray = np.empty(())
        self.group_person_decodes: Dict[str, np.ndarray] = {}
        self.person_img_info: Dict = {}
        self.info = {}
        self.detect_image:Image.Image = None

    def add_group_img(self, group_image: Image.Image):
        self.origin_image = group_image
        self.group_person_xywhs = self.detect_model.detect(
            group_image, conf=0.05)
        for item in self.group_person_xywhs:
            sub_face = get_expanded_sub_image(
                self.origin_image, item, expansion_factor=2)
            decoder: np.ndarray = self.recognition_model.decode(sub_face)
            decoder = decoder.reshape([-1])
            self.group_person_decodes[xywh2str(item)] = decoder

        logger.info(f"detect {len(self.group_person_xywhs)} persons")

    def add_person_img(self, person_images: dict[str, Image.Image]):
        for item in person_images:
            name = item
            person_image = person_images[name]
            logger.info(f"decoder for {item}")
            person_xywh = self.detect_model.detect(person_image)[0]
            person_face_image = crop_image_from_xywh(person_image, person_xywh)
            self.person_img_info[name] = self.recognition_model.decode(
                person_face_image)

    def analysis(self):

        group_candidate_dict = self.group_person_decodes

        person_candidate_dict = self.person_img_info.copy()
        unknow_group_face_dict = group_candidate_dict.copy()
        logger.info(f"person_candidate_dict is {len(person_candidate_dict)}")
        logger.info(f"unknow_group_face_dict is {len(unknow_group_face_dict)}")
       
        # recognization process 
        conf = 0.8
        while len(person_candidate_dict) > 0 and len(group_candidate_dict) > 0:
            temp_unknow_group_face = {}
            for xywh, group_face_coder in unknow_group_face_dict.items():

                name, score = query_data(
                    group_face_coder, person_candidate_dict)
                logger.info(f"score  {score}, conf is {conf}")
                if score > conf:
                    logger.info(f"best score name is {name}")
                    self.info[name] = {
                        "xywh": str2xywh(xywh),
                        "score": score
                    }
                    del person_candidate_dict[name]
                    del group_candidate_dict[xywh]
                    
                else:
                    temp_unknow_group_face[xywh]= group_face_coder
                    
                logger.info(f"person_candidate_dict is {len(person_candidate_dict)}")
                logger.info(f"group_candidate_dict is {len(group_candidate_dict)}")    
                logger.info(f"temp_unknow_group_face is {len(temp_unknow_group_face)}")   
                
            unknow_group_face_dict = temp_unknow_group_face.copy()
            conf = conf * conf
            
        self.info =  self.location_model(self.info)
        # emotion process
        for name,_ in self.info.items():
            xywh = self.info[name]["xywh"]
            group_face_image = get_expanded_sub_image(self.origin_image, xywh, expansion_factor=2)
            
            emotion, score = self.emotion_model.get_emotion(group_face_image)
            self.info[name]["origin_face"] = group_face_image
            self.info[name]["emotion"] = emotion
            self.info[name]["emotion_score"] = score
            
            matting_face = self.matting_model.matting(group_face_image)
            
            self.info[name]["matting_face"] = matting_face
            
        self.detect_image = self.origin_image.copy()
        
        # draw detect face for origin face 
        
        draw = ImageDraw.Draw(self.detect_image)

        for item in self.info:
            xywh = self.info[item]["xywh"]
            
            x, y, w, h = xywh

            x1 = int(x - (w/2))
            y1 = int(y - (h/2))
            x2 = int(x + (w/2))
            y2 = int(y + (h/2))
        # 绘制矩形
            draw.rectangle([(x1, y1), (x2, y2)], outline='red', width = 10)
        

def query_data(decoder: np.ndarray, candidate: Dict[str, np.ndarray]):
    decoder = decoder.reshape([-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score = 0.0
    decoder_tensor = torch.tensor(decoder, device=device)

    best_name = np.array(())
    for name, candidate_decoder in candidate.items():

        can_tensor = torch.tensor(
            candidate_decoder.reshape([-1]), device=device)
        candidate_score = (F.cosine_similarity(
            can_tensor, decoder_tensor, dim=0).cpu().item() + 1.0) / 2.0
        if candidate_score > score:
            score = candidate_score
            best_name = name

    return best_name, score
