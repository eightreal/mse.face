import torch
from PIL import Image
from typing import Tuple
from modelscope import pipeline, Tasks
from modelscope.outputs import OutputKeys
import numpy as np

# 检查是否支持cuda
device = "cuda" if torch.cuda.is_available else "cpu"

class Emotion:

    # 构造函数
    def __init__(self):

        # 将模型导入到pipline
        self.model = pipeline(
            Tasks.facial_expression_recognition, 'iic/cv_vgg19_facial-expression-recognition_fer', device=device)

    # 表情预测
    def get_emotion(self, image: Image) :
        """Return image sentiment and score

        Args:
            image (Image): Input picture
        """
        # 通过模型生成表情预测
        ret = self.model(image)

        # 取得分最高的预测表情下标
        label_idx = np.array(ret['scores']).argmax()

        # 得到并返回表情和预测得分
        label = ret['labels'][label_idx]
        score = ret['scores'][label_idx]
        return label,score

# 测试代码
# testEmotion = Emotion()
# person_dict = Image.open(MODEL_DIR+"/../data/person/01巴昊.png")
# img = testEmotion.get_emotion(person_dict)