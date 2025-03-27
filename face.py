"""Process processing class
"""
import sys
import os

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(''))
# 将当前目录添加到 sys.path 中
sys.path.append(current_dir)


import PIL
import numpy as np

import torch.nn.functional as F
import torch
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple
from loguru import logger


from face_package import crop_image_from_xywh, get_expanded_sub_image, xywh2str, str2xywh, FONT_PATH
from face_package.face_detect import Detect
from face_package.face_emotion import Emotion
from face_package.face_recognition import Recognition
from face_package.face_matting import Matting
from face_package.face_location import Location


# 创建处理类


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
        self.detect_image: Image.Image = None
        self.rows = []

    def add_group_img(self, group_image: Image.Image):
        self.origin_image = group_image
        # 获取所有检测人脸的xywh, 其代表中心点位置的xy, 以及宽度高度
        self.group_person_xywhs = self.detect_model.detect(
            group_image)
        # 获取合照中人脸的特征值
        for item in self.group_person_xywhs:
            # 将候选框的wh放大两倍获取再输入特征提取模型
            sub_face = get_expanded_sub_image(
                self.origin_image, item, expansion_factor=2)
            decoder: np.ndarray = self.recognition_model.decode(sub_face)
            # 转成一维向量
            decoder = decoder.reshape([-1])
            # 将获取的向量存储到类变量中
            self.group_person_decodes[xywh2str(item)] = decoder

        logger.info(f"detect {len(self.group_person_xywhs)} persons")

    def add_person_img(self, person_images: dict[str, Image.Image]):
        """
           此方法用于从输入的人物图像字典中提取人物面部特征并进行解码。

        Args:
           person_images (dict[str, Image.Image]): 一个字典，键为人物名称（字符串），值为对应的PIL Image对象。

        Returns:
           None: 此方法不返回值，而是将解码后的人物面部特征信息存储在实例变量 self.person_img_info 中。

         """
        self.person_img_info = {}
        for item in person_images:
            name = item
            person_image = person_images[name]
            logger.info(f"decoder for {item}")
            person_xywh = self.detect_model.detect(person_image)[0]
            person_face_image = get_expanded_sub_image(person_image, person_xywh,2)
            self.person_img_info[name] = self.recognition_model.decode(
                person_face_image)

    def analysis(self):
        """
           执行图像分析的主要流程，包括人脸识别、位置定位、情感分析和绘制检测结果。

           该方法首先复制人员和组的候选字典，然后通过循环进行人脸识别，将识别结果存入`self.info`中。
           接着，调用位置模型更新`self.info`和`self.rows`。之后，对每个人脸进行情感分析和抠图处理（当前代码中情感分析和抠图部分被注释掉）。
           最后，在原始图像上绘制检测到的人脸矩形框和对应的名称。

           :return: 无返回值，结果存储在类的属性中
           """
        # 复制组人员解码字典，避免修改原始数据
        group_candidate_dict = self.group_person_decodes.copy()
        # 复制人员图像信息字典，避免修改原始数据
        person_candidate_dict = self.person_img_info.copy()

        logger.info(f"person_candidate_dict is {len(person_candidate_dict)}")

        # recognization process
        while len(person_candidate_dict) > 0 and len(group_candidate_dict) > 0:
            best_score = 0.0
            best_name = "unknow"
            best_xywh = None
            # 遍历组候选字典中的每个键值对，键为xywh，值为组面部编码器
            for xywh, group_face_coder in group_candidate_dict.items():
                # 查询数据，获取名称和分数
                name, score = query_data(
                    group_face_coder, person_candidate_dict)
                # 如果当前分数大于最佳分数，则更新最佳分数、最佳名称和最佳xywh
                if score > best_score:
                    logger.info(f"best score name is {name}")
                    best_score=score
                    best_name = name
                    best_xywh = xywh

            # 将识别结果存入self.info字典中
            self.info[best_name] = {
                        "xywh": str2xywh(best_xywh),
                        "score": best_score
                    }
            logger.info(f"识别成功 {best_name}, {best_score}")
            # 从人员候选字典中删除已识别的人员
            del person_candidate_dict[best_name]
            # 从组候选字典中删除已匹配的组
            del group_candidate_dict[best_xywh]

        # 调用位置模型，更新self.info和self.rows
        self.info, self.rows = self.location_model(self.info)
        # 情感分析过程
        for name, _ in self.info.items():
            xywh = self.info[name]["xywh"]
            # 获取扩展的子图像
            group_face_image = get_expanded_sub_image(
                self.origin_image, xywh, expansion_factor=2)

            self.info[name]["origin_face"] = group_face_image
            self.info[name]["emotion"] = "emotion"
            self.info[name]["emotion_score"] = 0.0
            self.info[name]["matting_face"] = group_face_image

            emotion, score = self.emotion_model.get_emotion(group_face_image)
            self.info[name]["origin_face"] = group_face_image
            self.info[name]["emotion"] = emotion
            self.info[name]["emotion_score"] = score
            
            matting_face:Image.Image = self.matting_model.matting(group_face_image)
            self.info[name]["matting_face"] = matting_face
            
        # 复制原始图像，用于绘制检测结果
        self.detect_image = self.origin_image.copy()

        # draw detect face for origin face

        draw = ImageDraw.Draw(self.detect_image)
        # 遍历识别结果，在图像上绘制矩形框和名称
        for item in self.info:
            xywh = self.info[item]["xywh"]

            x, y, w, h = xywh

            x1 = int(x - (w/2))
            y1 = int(y - (h/2))
            x2 = int(x + (w/2))
            y2 = int(y + (h/2))
            # 设置字体
            font = PIL.ImageFont.truetype(
                # 或者 encoding = "unic" ,encoding="gbk"
                font=f"{FONT_PATH}", size=80, encoding="utf-8")
            # 绘制矩形
            draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=10)
            # 绘制名称
            draw.text((x1, y1-80), item, font=font, fill=(0, 0, 255))


def query_data(decoder: np.ndarray, candidate: Dict[str, np.ndarray]):
    """_summary_

    Args:
        decoder (np.ndarray): 当前合照人脸的特征值
        candidate (Dict[str, np.ndarray]): 候选的个人照人脸特征dict

    Returns:
        _type_: 最佳分数的 名称与分数
    """
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


def draw_detect(xywh, group_img: Image.Image):
    """根据xywh 在图片上绘制矩形

    Args:
        xywh (_type_): _description_
        group_img (Image.Image): _description_

    Returns:
        _type_: _description_
    """
    draw_img = group_img.copy()
    x, y, w, h = xywh
    x1 = int(x - (w/2))
    y1 = int(y - (h/2))
    x2 = int(x + (w/2))
    y2 = int(y + (h/2))
    # 绘制矩形
    draw = ImageDraw.Draw(draw_img)
    draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=20)
    return draw_img
