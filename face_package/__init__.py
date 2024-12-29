import sys
import os
from PIL import Image
from numpy import ndarray

# 获取当前文件所在的目录
_current_dir = os.path.dirname(__file__)

DATA_DIR = _current_dir + "/../data"
MODEL_DIR = _current_dir + "/../model"


def crop_image_from_xywh(image: Image.Image, xywhs: ndarray) -> Image.Image:

    x, y, w, h = xywhs

    x1 = int(x - (w/2))
    y1 = int(y - (h/2))
    x2 = int(x + (w/2))
    y2 = int(y + (h/2))

    cropped_img = image.crop((x1, y1, x2, y2))

    return cropped_img


def get_expanded_sub_image(image: Image.Image, xywh: ndarray, expansion_factor: float = 1.2):
    # 计算扩充后的宽度和高度
    xywh[2] = xywh[2] * expansion_factor
    xywh[3] = xywh[3] * expansion_factor
    # 计算原始区域中心

    return crop_image_from_xywh(image, xywh)
