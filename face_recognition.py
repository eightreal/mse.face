import numpy as np
import timm
from PIL import Image
from transformers import pipeline
import torch
import torch.nn.functional as F
from git.util import to_native_path


device =  torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

class Recognition:
    def __init__(self):

        self.model = timm.create_model(
            "hf_hub:gaunernst/vit_tiny_patch8_112.arcface_ms1mv3", pretrained=True).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.transform = timm.data.create_transform(
            **timm.data.resolve_data_config(self.model.pretrained_cfg))

    def decode(self, img: Image) -> torch.Tensor:

        # 使用预处理后的图像数据进行模型推理
        if img.mode != 'RGB':
            img = img.convert('RGB')

        pimage_tensor = self.transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            pimage_tensor = pimage_tensor.cuda()
        # 使用预处理后的图像数据进行模型推理
        embs = self.model(pimage_tensor)  # output shape (1, 512)
        embs = F.normalize(embs, dim=1)

        # print(embs)
        return embs.to(device=device)
