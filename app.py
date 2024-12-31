#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from PIL import Image
from loguru import logger
import numpy as np
import streamlit as st
import cv2
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import List, Dict

import config
from face_package.face_detect import Detect
from face_package.face_recognition import Recognition
from face_package.face_emotion import Emotion
from face_package.face_matting import Matting
from face import Face


if "face_handle" not in st.session_state:
    face_handle = Face()
    st.session_state["face_handle"] = face_handle

if "person_images" not in st.session_state:
    st.session_state["person_images"] = {}
if "group_image" not in st.session_state:
    st.session_state["group_image"] = None
if 'analyzer_disabled' not in st.session_state:
    st.session_state.analyzer_disabled = True


# setting page layout
st.set_page_config(
    page_title="合照人脸识别系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# main page heading

container = st.container(border=True)


def should_disable_analyzer():
    return st.session_state.get('analyzer_disabled', True)


def submit_click():
    st.session_state["analyzer_disabled"] = False


def analyzer():
    face_handle: Face = st.session_state["face_handle"]
    face_handle.add_person_img()
    origin_col, face_col = container.columns(2)

    face_handle.add_group_img( st.session_state["group_images_content"])
    face_handle.add_person_img( st.session_state["person_images"])
    face_handle.analysis()

    with origin_col:
        origin_col.image(
            face_handle.origin_image, caption='上传并读取的图像')
    with face_col:
        face_col.image(face_handle.detect_image)


# sidebar
with st.sidebar as side:

    # model options
    with st.form("person_image_form", clear_on_submit=True):
        group_img_file: UploadedFile = st.file_uploader(
            "上传合照文件", accept_multiple_files=False)
        person_image_files: List[UploadedFile] = st.file_uploader(
            "上传个人文件", accept_multiple_files=True)

        sd_col1, sd_col2 = st.columns(2)

        submitted = sd_col1.form_submit_button("提交", on_click=submit_click)
        analyzer_button = sd_col2.form_submit_button(
            "开始分析", disabled=should_disable_analyzer())
        if submitted:
            logger.info(group_img_file)
            st.session_state["group_images_content"] = Image.open(
                group_img_file)
            st.session_state["person_images"] = {}
            for person_img_file in person_image_files:
                name_without_extension = os.path.splitext(person_img_file.name)[0]
                
                st.session_state["person_images"][name_without_extension] = Image.open(
                    person_img_file)

        if analyzer_button:
            analyzer()
