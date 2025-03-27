
import torch
from PIL import Image
import cv2
from modelscope import pipeline, Tasks
from modelscope.outputs import OutputKeys

# 检查是否支持cuda
device = "cuda" if torch.cuda.is_available else "cpu"

class Matting:

    # 构造函数
    def __init__(self):

        # 将模型导入到pipline
        self.model = pipeline(
            Tasks.portrait_matting, 'iic/cv_unet_image-matting', device=device)

    # 图像切割
    def matting(self, image: Image) -> Image:
        """Enter an image and filter out its background

        Args:
            image (Image): input image
        """
        # 通过模型切割图像
        result = self.model(image)

        # 图像转化为image格式
        output_image = Image.fromarray(cv2.cvtColor(result[OutputKeys.OUTPUT_IMG],cv2.COLOR_BGRA2RGBA))

        # 返回image格式数据
        return output_image

# 测试代码
# testMatting = Matting()
# person_dict = Image.open(MODEL_DIR+"/../data/person/01巴昊.png")
# img = testMatting.matting(person_dict)
