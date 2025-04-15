# import streamlit as st
#
# st.title("文档矫正与问答")
# import streamlit as st
# from PIL import Image
# import os
#
#
# # 创建一个文件上传器
# uploaded_file = st.file_uploader("上传一张图像", type=["png", "jpg", "jpeg", "gif"])
#
# # 检查是否上传了文件
# if uploaded_file is not None:
#     # 显示上传的图像
#     try:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="上传的图像", use_column_width=True)
#
#         # 提供保存图像的选项
#         if st.button("保存图像到本地"):
#             # 创建保存目录（如果不存在）
#             save_dir = "uploads"
#             os.makedirs(save_dir, exist_ok=True)
#
#             # 保存图像
#             save_path = os.path.join(save_dir, uploaded_file.name)
#             with open(save_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
#             st.success(f"图像已保存到: {save_path}")
#
#     except Exception as e:
#         st.error(f"无法打开图像: {e}")

import os
import sys
import base64
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import ollama
import streamlit as st
from io import BytesIO
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


def main():
    st.title("图像处理与OCR应用")

    # 初始化模型
    model_path = os.path.join(r"E:\PythonProject1\UVDoc-main\myfile\visualize\app_data\models\unwrap_model\best_model.pkl")
    model = load_model(model_path)
    # yolo_model_path = os.path.join(ROOT, r"F:\work_doc_cls\yolo_train\runs\classify\train5\weights\best.pt")
    yolo_model_path = os.path.join(r"E:\PythonProject1\UVDoc-main\myfile\visualize\app_data\models\cls_model\best.pt")
    yolo_model = load_yolo_model(yolo_model_path)

    # 上传图片
    uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # 保存上传的图片
        original_img_path = os.path.join(r"E:\PythonProject1\UVDoc-main\myfile\visualize\app_data\log\temp.png")
        with open(original_img_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # 显示原始图片
        st.image(uploaded_file, caption="原始图片", use_column_width=True)

        # 模型选择
        model_folder = os.path.join(r"E:\PythonProject1\UVDoc-main\myfile\visualize\app_data\models\unwrap_model")
        model_files = [f for f in os.listdir(model_folder)]
        selected_model = st.selectbox("选择模型", model_files)
        selected_model_path = os.path.join(model_folder, selected_model)

        # 处理图片
        if st.button("处理图片"):
            try:
                processed_path = unwarp_img(selected_model_path, original_img_path, IMG_SIZE, model)
                st.image(processed_path, caption="处理后的图片", use_column_width=True)
                st.session_state.processed_img_path = processed_path
            except Exception as e:
                st.error(f"图片处理失败: {e}")

        # YOLO 检测
        if st.button("检测图片"):
            if yolo_model is not None:
                try:
                    results = yolo_model(original_img_path)
                    annotated_img = plot_results(cv2.imread(original_img_path), results)
                    st.image(annotated_img, caption="检测结果", use_column_width=True, channels="BGR")
                except Exception as e:
                    st.error(f"检测失败: {e}")

        # OCR 功能
        ocr_prompt = st.text_input("OCR 提示词", "output the complete content of the image，without explain of the text")
        if st.button("执行 OCR"):
            if "processed_img_path" in st.session_state:
                ocr_result = perform_ocr(ocr_prompt, st.session_state.processed_img_path)
                st.text_area("OCR 结果", ocr_result, height=200)

        # 保存图片
        if st.button("保存图片"):
            if "processed_img_path" in st.session_state:
                with open(st.session_state.processed_img_path, "rb") as f:
                    img_bytes = f.read()
                st.download_button(
                    label="下载处理后的图片",
                    data=img_bytes,
                    file_name="processed_image.png",
                    mime="image/png"
                )

        # 复制文字
        if st.button("复制文字"):
            if "processed_img_path" in st.session_state:
                ocr_result = perform_ocr(ocr_prompt, st.session_state.processed_img_path)
                js = f"navigator.clipboard.writeText('{ocr_result}');alert('已复制到剪贴板');"
                html = f'<img src="data:image/gif;base64,{base64.b64encode(b"").decode()}" alt="">' \
                       f'<script>{js}</script>'
                st.components.v1.html(html, height=0)
if __name__ == "__main__":
    main()