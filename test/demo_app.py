import argparse
import os

import cv2
import numpy as np
import torch

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from utils import IMG_SIZE, bilinear_unwarping, load_model


def unwarp_img(ckpt_path, img_path, img_size):
    """
    Unwarp a document image using the model from ckpt_path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # Load model
    model = load_model(ckpt_path)
    model.to(device)
    model.eval()

    # Load image
    print(f"图片路径：{img_path}")
    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    inp = torch.from_numpy(cv2.resize(img, img_size).transpose(2, 0, 1)).unsqueeze(0)

    # Make prediction
    inp = inp.to(device)
    point_positions2D, _ = model(inp)

    # Unwarp
    size = img.shape[:2][::-1]
    unwarped = bilinear_unwarping(
        warped_img=torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device),
        point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
        img_size=tuple(size),
    )
    unwarped = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # Save result
    unwarped_BGR = cv2.cvtColor(unwarped, cv2.COLOR_RGB2BGR)
    save_path = os.path.splitext(img_path)[0] + "_unwarp.png"
    cv2.imwrite(save_path, unwarped_BGR)
    return save_path


def select_image():
    global original_image, processed_image, original_label, processed_label, original_img_path  # 添加 original_img_path 到全局变量
    img_path = filedialog.askopenfilename()
    if img_path:
        original_img_path = img_path  # 将 img_path 赋值给 original_img_path
        # 读取图片并保存为副本
        original_image = Image.open(img_path).copy()
        # 计算缩放尺寸以保持图像比例
        width, height = original_image.size
        aspect_ratio = width / height
        new_height = 500
        new_width = int(new_height * aspect_ratio)
        original_image = original_image.resize((new_width, new_height), Image.LANCZOS)
        original_photo = ImageTk.PhotoImage(original_image)
        original_label.config(image=original_photo)
        original_label.image = original_photo


def process_image():
    global original_image, processed_image, original_label, processed_label
    if original_image:
        img_path = "temp_original.png"
        original_image.save(img_path)
        processed_path = unwarp_img(args.ckpt_path, img_path, IMG_SIZE)
        processed_image = Image.open(processed_path)
        # 计算缩放尺寸以保持图像比例
        width, height = processed_image.size
        aspect_ratio = width / height
        new_height = 500
        new_width = int(new_height * aspect_ratio)
        processed_image = processed_image.resize((new_width, new_height), Image.LANCZOS)
        processed_photo = ImageTk.PhotoImage(processed_image)
        processed_label.config(image=processed_photo)
        processed_label.image = processed_photo


def save_image():
    global processed_image, original_img_path  # 添加 original_img_path 到全局变量
    if processed_image:
        save_folder = r"E:\PythonProject1\UVDoc-main\myfile\app_img\output"  # 指定保存文件夹
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        original_img_name = os.path.splitext(os.path.basename(original_img_path))[0]
        new_img_name = f"{original_img_name}_flat.png"
        default_path = os.path.join(save_folder, new_img_name)
        processed_image.save(default_path)
        print(f"已保存 {default_path}")


import ollama  # 新增ocr模型导入


def perform_ocr():
    global original_img_path  # 添加 original_img_path 到全局变量
    if original_img_path:
        response = ollama.chat(
            model='minicpm-v:latest',
            messages=[{
                'role': 'user',
                'content': ocr_prompt.get(),  # 使用提示词输入框中的内容
                'images': [original_img_path]
            }]
        )
        print(response.message.content)
        # 显示OCR结果（这里假设在界面上显示）
        ocr_result_label.config(text=response.message.content)


if __name__ == "__main__":
    import torch

    print(torch.cuda.is_available())
    print(torch.version.cuda)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt-path", type=str, default=r"E:\PythonProject1\UVDoc-main\model\best_model.pkl", help="Path to the model weights as pkl."
    )
    parser.add_argument("--img-path", type=str,default=r"E:\PythonProject1\UVDoc-main\img\unwarp.png", help="Path to the document image to unwarp.")

    args = parser.parse_args()



    # 创建主窗口
    root = tk.Tk()
    root.title("Document Unwarping")
    root.geometry("1600x750")  # 调整窗口大小以适应第三列

    # 创建网格布局
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)  # 新增第三列

    # 添加图片显示区域
    original_label = tk.Label(root)
    original_label.grid(row=0, column=0, padx=5, pady=5)
    processed_label = tk.Label(root)
    processed_label.grid(row=0, column=1, padx=5, pady=5)

    # 添加OCR结果显示标签到第三列
    ocr_result_label = tk.Label(root, text="", wraplength=500)  # 设置wraplength以适应图片区域大小
    ocr_result_label.grid(row=0, column=2, padx=5, pady=5)

    # 减少列之间的间距
    root.grid_columnconfigure(0, weight=1, minsize=500)
    root.grid_columnconfigure(1, weight=1, minsize=500)
    root.grid_columnconfigure(2, weight=1, minsize=500)

    select_button = tk.Button(root, text="选择图片", command=select_image)
    select_button.grid(row=1, column=0, pady=5)

    save_button = tk.Button(root, text="保存图片", command=save_image)
    save_button.grid(row=1, column=1, pady=5)

    process_button = tk.Button(root, text="处理图片", command=process_image)
    process_button.grid(row=1, column=0, columnspan=2, pady=5)

    # 添加提示词输入框
    ocr_prompt = tk.Entry(root, width=50)
    ocr_prompt.grid(row=2, column=0, padx=5, pady=5)
    ocr_prompt.insert(0, "output the complete content of the image，without explain of the text")

    # 添加OCR按钮
    ocr_button = tk.Button(root, text="OCR", command=perform_ocr)
    ocr_button.grid(row=2, column=1, pady=5)

    root.mainloop()
