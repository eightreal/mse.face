# 集体照人脸识别代码


安装环境

```
pip install -r ./requirement.txt
```
请参照魔搭官方说明安装cv支持
https://www.modelscope.cn/docs/intro/quickstart

运行

```
streamlit run ./app.py
```

项目结构说明

app.py: streamlit app
face.py: 流程处理类
face_test.ipynb: 为我们流程的测试ipynb

face_package.face_detect.py 使用yolo检测人脸
+ 使用yolov11模型并在人脸数据集中进行微调，引用github地址如下 https://github.com/akanametov/yolo-face


face_package.face_emotion.py 使用魔搭模型检测情绪

face_package.face_location.py 根据xy值获取行列值
+ 使用kmeans聚类方法


face_package.face_matting.py 获取matting图像

face_package.face_recognition 人脸比对模型
- 魔搭链接如下： https://www.modelscope.cn/models/iic/cv_ir50_face-recognition_arcface
- 该模型主要贡献再提出了角度距离
