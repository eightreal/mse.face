#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     app.py
   @Author:        ang
   @Date:          2023/9/12
   @Description:
-------------------------------------------------
"""
from pathlib import Path
from PIL import Image
from loguru import logger
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_drawable_canvas import st_canvas
from typing import List

import config
from face_location import Location
# from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

analyzer_button = None
export_result_button = None
group_img = None
group_img_file: UploadedFile = None 
group_img_content = None
person_img_files: List[UploadedFile] = None
detect_model_type=None
recognition_model_type=None
emotion_model_type=None

# setting page layout
st.set_page_config(
    page_title="合照人脸识别系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# main page heading
st.title("合照人脸识别系统")

# sidebar
st.sidebar.header("模型配置选择")

# model options


detect_model_type = st.sidebar.selectbox(
    "选择人脸检测模型",
    config.DETECTION_MODEL_LIST)

recognition_model_type = st.sidebar.selectbox(
    "选择人脸识别模型",
    config.RECOGNITION_MODEL_LIST)

emotion_model_type = st.sidebar.selectbox(
    "选择情绪识别模型",
    config.EMOTION_MODEL_LIST)

confidence = float(st.sidebar.slider(
    "置信度", 30, 100, 50)) / 100

if detect_model_type:
    model_path = Path(config.MODEL_PATH, str(detect_model_type))
    

group_img_file: UploadedFile = st.sidebar.file_uploader(
    "上传合照文件", accept_multiple_files=False)

person_image_file: UploadedFile = st.sidebar.file_uploader(
    "上传个人文件", accept_multiple_files=True)



if group_img_file:
    logger.info(group_img_file)
    group_img = group_img_file

if group_img:
    group_img_content = Image.open(group_img)
    st.write("group image is ")
    st.image(group_img_content, caption="上传的图片", use_container_width=True)
    analyzer_button = st.button("开始分析")
    
if analyzer_button:
    location_model = Location(model_path=model_path)
    # location_model.detect()



