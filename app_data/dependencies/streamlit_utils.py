
import os
import sys
import base64
import cv2
import numpy as np

import torch
from ultralytics import YOLO
import ollama
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append(r'E:\PythonProject1\UVDoc-main\myfile\visualize\app_data\dependencies')
from utils import IMG_SIZE, bilinear_unwarping, load_model

# ROOT = os.path.dirname(os.path.abspath(__file__))


def unwarp_img(ckpt_path, img_path, img_size, model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    if model is None:
        model = load_model(ckpt_path)
    model.to(device)
    model.eval()

    # 加载图片
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    inp = torch.from_numpy(cv2.resize(img, img_size).transpose(2, 0, 1)).unsqueeze(0)

    # 调用模型预测
    inp = inp.to(device)
    point_positions2D, _ = model(inp)

    # 展平
    size = img.shape[:2][::-1]
    unwarped = bilinear_unwarping(
        warped_img=torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device),
        point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
        img_size=tuple(size),
    )
    unwarped = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # 保存结果
    unwarped_BGR = cv2.cvtColor(unwarped, cv2.COLOR_RGB2BGR)
    save_path = os.path.splitext(img_path)[0] + "_unwarp.png"
    cv2.imwrite(save_path, unwarped_BGR)
    return save_path


def perform_ocr(ocr_prompt, processed_img_path):
    try:
        if ocr_prompt:
            with open(processed_img_path, "rb") as f:
                image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode("utf-8")

            response = ollama.chat(
                model='granite3.2-vision',
                messages=[{
                    'role': 'user',
                    'content': ocr_prompt,
                    'images': [image_b64]
                }]
            )

            return response.message.content
    except Exception as e:
        return f"OCR Error: {e}"


def load_yolo_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"YOLO模型加载失败: {e}")
        return None


def plot_results(cv_img, results):
    # 获取分类结果
    probs = results[0].probs  # 获取概率结果
    top1_label = results[0].names[probs.top1]  # 获取最高概率类别
    top1_conf = probs.top1conf.item()  # 获取置信度

    # 在图像中心添加文字
    text = f"{top1_label} "
    font_scale = 2
    thickness = 4

    # 计算文字位置
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (cv_img.shape[1] - text_size[0]) // 2
    text_y = (cv_img.shape[0] + text_size[1]) // 2

    # 添加背景
    cv2.rectangle(cv_img,
                  (text_x - 10, text_y - text_size[1] - 10),
                  (text_x + text_size[0] + 10, text_y + 10),
                  (255, 255, 255), -1)

    if top1_conf > 0.7:
        cv2.putText(cv_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    else:
        cv2.putText(cv_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness - 1)
    return cv_img
