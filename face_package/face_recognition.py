import numpy as np
from PIL import Image
from modelscope import pipeline, Tasks
from modelscope.outputs import OutputKeys
import torch

device = "cuda" if torch.cuda.is_available else "cpu"


class Recognition:
    def __init__(self):

        self.model = pipeline(
            Tasks.face_recognition, 'damo/cv_ir50_face-recognition_arcface', device=device)

    def decode(self, img: Image) -> np.ndarray:

        # 使用预处理后的图像数据进行模型推理
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')

        # pimage_tensor = self.transform(img).unsqueeze(0)
        # if torch.cuda.is_available():
        #     pimage_tensor = pimage_tensor.cuda()
        # # 使用预处理后的图像数据进行模型推理
        # embedding = self.model(pimage_tensor)  # output shape (1, 512)
        # embedding = F.normalize(embedding, dim=1)
        embedding_np = self.model(img)[OutputKeys.IMG_EMBEDDING]
        return embedding_np
