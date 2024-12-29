#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from PIL import Image
from loguru import logger
import numpy as np
import streamlit as st
import cv2
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import List,Dict

import config
from face_location import Location
from face_recognition import  Recognition
from face_emotion import Emotion
from face_matting import Matting
from face import Face

location_model = Location()
recognition_model = Recognition()
emotion_model = Emotion()
matting_model = Matting()
face_handle = Face(location_model, emotion_model, recognition_model,matting_model)


analyzer_button = None
export_result_button = None
group_img = None
group_img_file: UploadedFile = None
group_img_content = None
person_img_files: List[UploadedFile] = None
person_dict: Dict[str, Image] = None
detect_model_type = None
recognition_model_type = None
emotion_model_type = None

st.session_state["group_imag"] = None
st.session_state["person_images"] = []


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



if detect_model_type:
    model_path = Path(config.MODEL_PATH, str(detect_model_type))


group_img_file: UploadedFile = st.sidebar.file_uploader(
    "上传合照文件", accept_multiple_files=False)


with st.sidebar.form("person_image_form", clear_on_submit=True):
    person_image_file = st.file_uploader(
        "上传个人照文件", accept_multiple_files=True)
    if person_image_file:
        person_dict = {}
        for item in person_image_file:
            file_path = Path(item.name)
            file_name_without_ext = file_path.stem
            person_dict[file_name_without_ext] = Image.open(item)
        logger.info(f"person dict len {len(person_dict)}")
        face_handle.add_person_img(person_dict)

    submitted = st.form_submit_button("提交")
    if submitted:
        logger.info((len(person_dict)))


if group_img_file:
    logger.info(group_img_file)
    group_img_content = Image.open(group_img_file)
    face_handle.add_group_img(group_img_content)
    st.write("group image is ")
    # st.image(group_img_content, caption="上传的图片", use_container_width=True)
    st.image(group_img_content, channels="BGR", caption='上传并读取的图像')


    analyzer_button = st.button("开始分析")
    
    if analyzer_button:
        with st.spinner("正在处理，请稍候..."):

            face_handle.detect()
        st.success("Done!")


