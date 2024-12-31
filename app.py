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
from face import Face, draw_detect

analyzer_button = None
search_click = None
name = None


if "face_handle" not in st.session_state:
    face_handle = Face()
    st.session_state["face_handle"] = face_handle

if "person_images" not in st.session_state:
    st.session_state["person_images"] = {}
if "group_image" not in st.session_state:
    st.session_state["group_image"] = None
if 'analyzer_disabled' not in st.session_state:
    st.session_state.analyzer_disabled = True
if "analysised" not in st.session_state:
    st.session_state.analysised = False

# setting page layout
st.set_page_config(
    page_title="合照人脸识别系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


if st.session_state.analysised:
    face_handle: Face = st.session_state["face_handle"]
    origin_col, face_col = st.tabs(["原始图片", "检测图片"])

    origin_col.image(
        face_handle.origin_image, caption='上传并读取的图像')

    face_col.image(face_handle.detect_image)

    name_col, search_col = st.columns(2, vertical_alignment='bottom')

    name = name_col.selectbox(
        "输入查询名称",
        options=list(face_handle.info.keys()),  # 也可以用元组
        index=1
    )
    search_click = search_col.button("search")


# main page heading


@st.dialog("搜索结果", width="large")
def search_func(name):
    st.write(f"{name} 的详细信息")
    face_handle: Face = st.session_state["face_handle"]
    logger.info(face_handle.info
                )
    info = face_handle.info[name]
    xywh = info["xywh"]
    group_img = draw_detect(xywh, group_img=face_handle.origin_image)
    origin_face = info["origin_face"]
    emotion = info["emotion"]
    emotion_score = info["emotion_score"]
    matting_face = info["matting_face"]
    col1, col2 = st.columns(2, vertical_alignment="center")
    tab1, tab2 = col1.tabs(["原始图片", "抠图图片"])
    tab1.image(origin_face)
    tab2.image(matting_face)
    col2.write(f"姓名 {name}")
    col2.write(f"情绪分数： {emotion_score}")
    col2.write(f"情绪： {emotion}")
    col2.write(f"行： {info["row"]}")
    col2.write(f"列：{info["col"]}")
    st.image(group_img)


def should_disable_analyzer():
    return st.session_state.get('analyzer_disabled', True)


def submit_click():
    st.session_state["analyzer_disabled"] = False


def analyzer():
    face_handle: Face = st.session_state["face_handle"]
    with st.spinner():
        face_handle.add_group_img(st.session_state["group_images_content"])
        face_handle.add_person_img(st.session_state["person_images"])
        face_handle.analysis()
    st.session_state.analysised = True
    st.session_state.analyzer_disabled = True

    

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
            "开始分析", disabled=should_disable_analyzer(), on_click=analyzer)
        if submitted:
            logger.info(group_img_file)
            st.session_state["group_images_content"] = Image.open(
                group_img_file)
            st.session_state["person_images"] = {}
            for person_img_file in person_image_files:
                name_without_extension = os.path.splitext(
                    person_img_file.name)[0]

                st.session_state["person_images"][name_without_extension] = Image.open(
                    person_img_file)

if search_click:
    search_func(name)
