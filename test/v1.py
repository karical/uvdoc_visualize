import argparse
import os
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, \
    QFileDialog, QWidget, QComboBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import cv2
import numpy as np

from PIL import Image

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
    img_path, _ = QFileDialog.getOpenFileName()
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
        original_photo = QPixmap(img_path)
        original_label.setPixmap(original_photo)
        original_label.setScaledContents(True)


def process_image():
    global original_image, processed_image, original_label, processed_label
    if original_image:
        img_path = "temp_original.png"
        original_image.save(img_path)
        processed_path = unwarp_img(args.ckpt_path, img_path, IMG_SIZE)
        processed_image = Image.open(processed_path)
        # 计算缩放尺寸,保持图像比例显示
        width, height = processed_image.size
        aspect_ratio = width / height
        new_height = 500
        new_width = int(new_height * aspect_ratio)
        processed_image = processed_image.resize((new_width, new_height), Image.LANCZOS)
        processed_photo = QPixmap(processed_path)
        processed_label.setPixmap(processed_photo)
        processed_label.setScaledContents(True)


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


def perform_ocr(ocr_result_label, ocr_prompt, main_window):
    try:
        if ocr_prompt.text():  # 确保提示词输入框中有内容
            # 添加调试信息
            print(f"OCR Prompt: {ocr_prompt.text()}")
            print(f"Original Image Path: {main_window.original_img_path}")

            response = ollama.chat(
                model='minicpm-v:latest',
                messages=[{
                    'role': 'user',
                    'content': ocr_prompt.text(),  # 使用提示词输入框中的内容
                    'images': [main_window.original_img_path]  # 使用 main_window 的 original_img_path
                }]
            )
            print(f"OCR Response: {response.message.content}")
            # 显示OCR结果（这里假设在界面上显示）
            ocr_result_label.setText(response.message.content)
    except Exception as e:
        print(f"OCR Error: {e}")
        ocr_result_label.setText(f"OCR Error: {e}，模型未加载")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("main window")
        self.resize(1600, 750)  # 调整窗口大小以适应图片显示

        # 初始化图片路径和图片对象
        self.original_img_path = ""
        self.original_image = None
        self.processed_image = None
        self.model_path = r"E:\PythonProject1\UVDoc-main\model\best_model.pkl"  # 默认模型路径

        # 添加图片显示区域
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 2px solid black; padding: 5px;")
        self.original_label.setText("原始图片")

        self.processed_label = QLabel()
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setStyleSheet("border: 2px solid black; padding: 5px;")
        self.processed_label.setText("处理后图片")

        # 添加OCR结果显示标签到主窗口
        self.ocr_result_label = QLabel()
        self.ocr_result_label.setWordWrap(True)  # 设置wraplength以适应图片区域大小
        self.ocr_result_label.setAlignment(Qt.AlignCenter)
        self.ocr_result_label.setStyleSheet("border: 2px solid black; padding: 5px;")
        self.ocr_result_label.setText("OCR结果")

        # 将所有部件添加到主布局
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.original_label)
        main_layout.addWidget(self.processed_label)
        main_layout.addWidget(self.ocr_result_label)

        # 设置主窗口的中心部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)


class ControlWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        self.setWindowTitle("控制面板")
        self.resize(300, 300)
        self.main_window = main_window

        # 添加模型选择下拉框
        self.model_combo = QComboBox()
        model_folder = r"E:\PythonProject1\UVDoc-main\myfile\app_data\model"
        model_files = [f for f in os.listdir(model_folder)]  # 修改为包含所有文件
        self.model_combo.addItems(model_files)
        self.model_combo.currentIndexChanged.connect(self.update_model_path)

        # 添加按钮
        self.select_button = QPushButton("选择图片")
        self.select_button.clicked.connect(self.select_image)
        self.save_button = QPushButton("保存图片")
        self.save_button.clicked.connect(self.save_image)
        self.process_button = QPushButton("处理图片")
        self.process_button.clicked.connect(self.process_image)

        # 添加提示词输入框
        self.ocr_prompt = QLineEdit()
        self.ocr_prompt.setPlaceholderText("output the complete content of the image，without explain of the text")
        self.ocr_prompt.setText("output the complete content of the image，without explain of the text")

        # 添加OCR按钮
        self.ocr_button = QPushButton("OCR")
        self.ocr_button.clicked.connect(
            lambda: perform_ocr(self.main_window.ocr_result_label, self.ocr_prompt, self.main_window))

        # 添加复制按钮
        self.copy_button = QPushButton("复制文字")
        self.copy_button.clicked.connect(self.copy_text)

        # 创建按钮布局
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.model_combo)  # 模型选择下拉框
        button_layout.addWidget(self.ocr_prompt)
        button_layout.addWidget(self.ocr_button)
        button_layout.addWidget(self.copy_button)

        # 设置浮动窗口的中心部件
        control_central_widget = QWidget()
        control_central_widget.setLayout(button_layout)
        self.setCentralWidget(control_central_widget)

    def update_model_path(self):
        selected_model = self.model_combo.currentText()
        model_folder = r"E:\PythonProject1\UVDoc-main\myfile\app_data\model"
        self.main_window.model_path = os.path.join(model_folder, selected_model)

    def select_image(self):
        img_path, _ = QFileDialog.getOpenFileName()
        if img_path:
            self.main_window.original_img_path = img_path  # 将 img_path 赋值给 main_window 的 original_img_path
            # 读取图片并保存为副本
            self.main_window.original_image = Image.open(img_path).copy()
            # 计算缩放尺寸以保持图像比例
            width, height = self.main_window.original_image.size
            aspect_ratio = width / height
            new_height = 500
            new_width = int(new_height * aspect_ratio)
            self.main_window.original_image = self.main_window.original_image.resize((new_width, new_height),
                                                                                     Image.LANCZOS)
            original_photo = QPixmap(img_path)
            self.main_window.original_label.setPixmap(original_photo)
            self.main_window.original_label.setScaledContents(True)

    def process_image(self):
        if self.main_window.original_image:
            img_path = "temp_original.png"
            self.main_window.original_image.save(img_path)
            # 使用选中的模型路径
            processed_path = unwarp_img(self.main_window.model_path, img_path, IMG_SIZE)
            self.main_window.processed_image = Image.open(processed_path)
            # 计算缩放尺寸以保持图像比例
            width, height = self.main_window.processed_image.size
            aspect_ratio = width / height
            new_height = 500
            new_width = int(new_height * aspect_ratio)
            self.main_window.processed_image = self.main_window.processed_image.resize((new_width, new_height),
                                                                                       Image.LANCZOS)
            processed_photo = QPixmap(processed_path)
            self.main_window.processed_label.setPixmap(processed_photo)
            self.main_window.processed_label.setScaledContents(True)

    def save_image(self):
        if self.main_window.processed_image:
            save_path, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "PNG Files (*.png);;All Files (*)")
            if save_path:
                self.main_window.processed_image.save(save_path)
                print(f"已保存 {save_path}")

    def copy_text(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.main_window.ocr_result_label.text())


if __name__ == "__main__":
    import torch

    print(torch.cuda.is_available())
    print(torch.version.cuda)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt-path", type=str, default=r"E:\PythonProject1\UVDoc-main\model\best_model.pkl",
        help="Path to the model weights as pkl."
    )
    parser.add_argument("--img-path", type=str, default=r"E:\PythonProject1\UVDoc-main\img\unwarp.png",
                        help="Path to the document image to unwarp.")

    args = parser.parse_args()

    app = QApplication(sys.argv)

    # 创建主窗口和控制窗口
    root = MainWindow()
    control_window = ControlWindow(root)

    root.show()
    control_window.show()
    sys.exit(app.exec_())
#######################################################################################

import os
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, \
    QFileDialog, QWidget, QComboBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import cv2
import numpy as np
from PIL import Image
from utils import IMG_SIZE, bilinear_unwarping, load_model

ROOT = os.path.dirname(os.path.abspath(__file__))


# print(f"当前路径：{ROOT}")

def unwarp_img(ckpt_path, img_path, img_size, model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    if model is None:
        model = load_model(ckpt_path)
    model.to(device)
    model.eval()

    # 加载图片
    print(f"图片路径：{img_path}")
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


import ollama  # 新增ocr模型导入


def perform_ocr(ocr_result_label, ocr_prompt, main_window):
    try:
        if ocr_prompt.text():
            # 添加调试信息
            print(f"OCR Prompt: {ocr_prompt.text()}")
            print(f"Original Image Path: {main_window.original_img_path}")

            response = ollama.chat(
                model='minicpm-v:latest',
                messages=[{
                    'role': 'user',
                    'content': ocr_prompt.text(),
                    'images': [main_window.original_img_path]  # 使用 main_window 的 original_img_path
                }]
            )
            print(f"OCR Response: {response.message.content}")
            # 显示OCR结果
            ocr_result_label.setText(response.message.content)
    except Exception as e:
        print(f"OCR Error: {e}")
        ocr_result_label.setText(f"OCR Error: {e}，模型未加载")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("main window")
        self.resize(1600, 750)  # 调整窗口大小以适应图片显示

        # 初始化图片路径和图片对象
        self.original_img_path = ""
        self.original_image = None
        self.processed_image = None
        # 默认模型路径
        self.model_path = os.path.join(ROOT, r"app_data\model\best_model.pkl")
        self.model = None  # 添加模型属性

        # 添加图片显示区域
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 2px solid black; padding: 5px;")
        self.original_label.setText("原始图片")

        self.processed_label = QLabel()
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setStyleSheet("border: 2px solid black; padding: 5px;")
        self.processed_label.setText("处理后图片")

        # 添加OCR结果显示标签到主窗口
        self.ocr_result_label = QLabel()
        self.ocr_result_label.setWordWrap(True)  # 设置wraplength以适应图片区域大小
        self.ocr_result_label.setAlignment(Qt.AlignCenter)
        self.ocr_result_label.setStyleSheet("border: 2px solid black; padding: 5px;")
        self.ocr_result_label.setText("OCR结果")

        # 将所有部件添加到主布局
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.original_label)
        main_layout.addWidget(self.processed_label)
        main_layout.addWidget(self.ocr_result_label)

        # 设置主窗口的中心部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def set_image_to_label(self, label, image_path):
        pixmap = QPixmap(image_path)
        # 保持图片的宽高比，并缩放到适应label的大小
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        label.setScaledContents(True)


class ControlWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        self.setWindowTitle("控制面板")
        self.resize(300, 300)
        self.main_window = main_window

        # 添加模型选择下拉框
        self.model_combo = QComboBox()
        model_folder = os.path.join(ROOT, r"app_data\model")
        model_files = [f for f in os.listdir(model_folder)]  # 修改为包含所有文件
        self.model_combo.addItems(model_files)
        self.model_combo.currentIndexChanged.connect(self.update_model_path)

        # 添加按钮
        self.select_button = QPushButton("选择图片")
        self.select_button.clicked.connect(self.select_image)
        self.save_button = QPushButton("保存图片")
        self.save_button.clicked.connect(self.save_image)
        self.process_button = QPushButton("处理图片")
        self.process_button.clicked.connect(self.process_image)

        # 添加提示词输入框
        self.ocr_prompt = QLineEdit()
        self.ocr_prompt.setPlaceholderText("output the complete content of the image，without explain of the text")
        self.ocr_prompt.setText("output the complete content of the image，without explain of the text")

        # 添加OCR按钮
        self.ocr_button = QPushButton("OCR")
        self.ocr_button.clicked.connect(
            lambda: perform_ocr(self.main_window.ocr_result_label, self.ocr_prompt, self.main_window))

        # 添加复制按钮
        self.copy_button = QPushButton("复制文字")
        self.copy_button.clicked.connect(self.copy_text)

        # 创建按钮布局
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.model_combo)  # 模型选择下拉框
        button_layout.addWidget(self.ocr_prompt)
        button_layout.addWidget(self.ocr_button)
        button_layout.addWidget(self.copy_button)

        # 设置浮动窗口的中心部件
        control_central_widget = QWidget()
        control_central_widget.setLayout(button_layout)
        self.setCentralWidget(control_central_widget)

    def update_model_path(self):
        selected_model = self.model_combo.currentText()
        model_folder = os.path.join(ROOT, r"app_data\model")
        self.main_window.model_path = os.path.join(model_folder, selected_model)
        # 清空处理后的图片
        self.main_window.processed_image = None
        self.main_window.set_image_to_label(self.main_window.processed_label, "")
        # 重新加载模型
        try:
            if self.main_window.model is not None:
                del self.main_window.model  # 删除旧模型
                torch.cuda.empty_cache()  # 清空CUDA缓存
            self.main_window.model = load_model(self.main_window.model_path)
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.main_window.model = None
            self.main_window.ocr_result_label.setText(f"模型加载失败: {e}")

    def select_image(self):
        # 设置默认文件夹路径
        select_folder = os.path.join(ROOT, r"app_data\benchmark")
        img_path, _ = QFileDialog.getOpenFileName(self, "选择图片", select_folder)
        if img_path:
            self.main_window.original_img_path = img_path  # 将 img_path 赋值给 main_window 的 original_img_path
            # 读取图片并保存为副本
            self.main_window.original_image = Image.open(img_path).copy()
            # 设置图片到label
            self.main_window.set_image_to_label(self.main_window.original_label, img_path)

    def process_image(self):
        if self.main_window.original_image:
            img_path = "app_data/log/temp.png"
            self.main_window.original_image.save(img_path)
            # 使用选中的模型路径
            processed_path = unwarp_img(self.main_window.model_path, img_path, IMG_SIZE, self.main_window.model)
            self.main_window.processed_image = Image.open(processed_path)
            # 设置图片到label
            self.main_window.set_image_to_label(self.main_window.processed_label, processed_path)

    def save_image(self):
        save_folder = os.path.join(ROOT, r"app_data\saved_image")
        if self.main_window.processed_image:
            save_path, _ = QFileDialog.getSaveFileName(self, "保存图片", save_folder,
                                                       "PNG Files (*.png);;All Files (*)")
            if save_path:
                self.main_window.processed_image.save(save_path)
                print(f"已保存 {save_path}")

    def copy_text(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.main_window.ocr_result_label.text())


if __name__ == "__main__":
    import torch

    print(torch.cuda.is_available())
    print(torch.version.cuda)
    app = QApplication(sys.argv)

    # 创建主窗口和控制窗口
    root = MainWindow()
    control_window = ControlWindow(root)

    root.show()
    control_window.show()
    sys.exit(app.exec_())
