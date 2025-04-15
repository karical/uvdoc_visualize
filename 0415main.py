import sys
import asyncio
import base64
import numpy as np
from ultralytics import YOLO
import ollama
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import cv2
import streamlit as st
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
os.environ["STREAMLIT_SERVER_ENABLE_WATCH_DOG"] = "false"
ROOT = os.path.dirname(os.path.abspath(__file__))
################导入设置##########################

IMG_SIZE = [488, 712]
GRID_SIZE = [45, 31]


def load_model(ckpt_path):
    """
    Load UVDocnet model.
    """
    model = UVDocnet(num_filter=32, kernel_size=5)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state"])
    return model


def bilinear_unwarping(warped_img, point_positions, img_size):
    """
    Utility function that unwarps an image.
    Unwarp warped_img based on the 2D grid point_positions with a size img_size.
    Args:
        warped_img  :       torch.Tensor of shape BxCxHxW (dtype float)
        point_positions:    torch.Tensor of shape Bx2xGhxGw (dtype float)
        img_size:           tuple of int [w, h]
    """
    upsampled_grid = F.interpolate(
        point_positions, size=(img_size[1], img_size[0]), mode="bilinear", align_corners=True
    )
    unwarped_img = F.grid_sample(warped_img, upsampled_grid.transpose(1, 2).transpose(2, 3), align_corners=True)

    return unwarped_img
################导入设置##########################
# 添加自定义CSS样式
st.markdown("""
<style>
    .stApp { background-color: #f5f5f5; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stDownloadButton>button {
        background-color: #008CBA !important;
    }
    .result-box {
        padding: 1rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# from app_data.dependencies.utils import IMG_SIZE, bilinear_unwarping, load_model
# from app_data.dependencies.streamlit_utils import load_yolo_model, unwarp_img, plot_results, perform_ocr


def main():
    st.title("智能文档处理系统")
    st.markdown("---")

    # 初始化模型
    with st.spinner("正在初始化模型..."):
        # model_path = os.path.join(
        #     r"E:\PythonProject1\UVDoc-main\myfile\visualize\app_data\models\unwrap_model\best_model.pkl")
        model_path = os.path.join(ROOT,"app_data/models/unwrap_model/best_model.pkl")
        model = load_model(model_path)
        # yolo_model_path = os.path.join(
        #     r"E:\PythonProject1\UVDoc-main\myfile\visualize\app_data\models\cls_model\best.pt")
        yolo_model_path = os.path.join(ROOT,"app_data/models/cls_model/best.pt")
        yolo_model = load_yolo_model(yolo_model_path)

    # 侧边栏配置
    with st.sidebar:
        st.header("配置选项")
        model_folder = os.path.join(ROOT,"app_data/models/unwrap_model")
        model_files = [f for f in os.listdir(model_folder)]
        selected_model = st.selectbox("选择校正模型", model_files, index=0)
        selected_model_path = os.path.join(model_folder, selected_model)

        st.markdown("---")
        ocr_prompt = st.text_input("OCR提示词",
                                   value="请提取并输出图像中的所有文字，包括手写文字、印刷文字以及可能存在的多语言文字。输出时请确保文字的完整性和可读性。如果图像中存在模糊或难以识别的文字，请尽量提供可能的推测结果",
                                   help="输入给OCR模型的提示指令")
        st.markdown("---")
        st.info("使用说明：\n1. 上传图片\n2. 选择处理功能\n3. 查看结果")

    # 主内容区域
    uploaded_file = st.file_uploader("上传文档图片", type=["png", "jpg", "jpeg"],
                                     help="支持PNG/JPG/JPEG格式")

    if uploaded_file is not None:
        # 保存上传的图片
        original_img_path = os.path.join(ROOT,"app_data/log/temp.png")
        with open(original_img_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # 三列布局容器
        col1, col2, col3 = st.columns(3)

        # 第一列：原始图片
        with col1:
            st.subheader("原始图片")
            st.image(uploaded_file,
                     use_column_width=True,
                     caption="上传的原始文档图片")



        # 第三列：检测结果
        with col2:
            st.subheader("检测分析")
            if "annotated_img" in st.session_state:
                st.image(st.session_state.annotated_img,
                         channels="BGR",
                         use_column_width=True,
                         caption="文档检测结果",
                         output_format="PNG")
                # 分析详情面板
                # with st.expander("📊 检测详情"):
                #     st.code("检测置信度：92%\n分类结果：正式合同文档")
            else:
                st.info("点击下方按钮进行文档检测")

        # 第二列：校正结果
        with col3:
            st.subheader("校正结果")
            if "processed_img_path" in st.session_state:
                st.image(st.session_state.processed_img_path,
                         use_column_width=True,
                         caption="文档校正结果")
                # 下载按钮
                with open(st.session_state.processed_img_path, "rb") as f:
                    img_bytes = f.read()
                st.download_button(
                    label="⬇️ 下载校正结果",
                    data=img_bytes,
                    file_name="corrected_document.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                st.info("点击下方按钮进行文档校正")

        # 功能按钮区域
        st.markdown("---")
        btn_col1, btn_col2, _ = st.columns([1, 1, 4])
        with btn_col1:
            if st.button("文档校正", help="进行文档几何校正处理"):
                with st.spinner("正在校正文档..."):
                    try:
                        processed_path = unwarp_img(selected_model_path, original_img_path, IMG_SIZE, model)
                        st.session_state.processed_img_path = processed_path
                        st.rerun()
                    except Exception as e:
                        st.error(f"校正失败: {str(e)}")

        with btn_col2:
            if st.button("文档检测", help="进行文档分类检测"):
                if yolo_model is not None:
                    with st.spinner("正在分析文档..."):
                        try:
                            results = yolo_model(original_img_path)
                            annotated_img = plot_results(cv2.imread(original_img_path), results)
                            st.session_state.annotated_img = annotated_img
                            st.rerun()
                        except Exception as e:
                            st.error(f"检测失败: {str(e)}")

        # # OCR功能区域
        # st.markdown("---")
        # if st.button("📖 执行OCR识别", use_container_width=True):
        #     if "processed_img_path" in st.session_state:
        #         with st.spinner("正在识别文字..."):
        #             ocr_result = perform_ocr(ocr_prompt, st.session_state.processed_img_path)
        #             st.session_state.ocr_result = ocr_result
        #             st.rerun()
        #
        # if "ocr_result" in st.session_state:
        #     st.subheader("OCR识别结果")
        #     with st.container():
        #         st.markdown(f'<div class="result-box">{st.session_state.ocr_result}</div>',
        #                     unsafe_allow_html=True)
        #         if st.button("📋 复制到剪贴板", use_container_width=True):
        #             js = f"""
        #             navigator.clipboard.writeText(`{st.session_state.ocr_result}`);
        #             setTimeout(() => alert("复制成功！"), 100);
        #             """
        #             html = f'<script>{js}</script>'
        #             st.components.v1.html(html, height=0)

def conv3x3(in_channels, out_channels, kernel_size, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def dilated_conv_bn_act(in_channels, out_channels, act_fn, BatchNorm, dilation):
    model = nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            bias=False,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        ),
        BatchNorm(out_channels),
        act_fn,
    )
    return model


def dilated_conv(in_channels, out_channels, kernel_size, dilation, stride=1):
    model = nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
        )
    )
    return model


class ResidualBlockWithDilation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        BatchNorm,
        kernel_size,
        stride=1,
        downsample=None,
        is_activation=True,
        is_top=False,
    ):
        super(ResidualBlockWithDilation, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.is_activation = is_activation
        self.is_top = is_top
        if self.stride != 1 or self.is_top:
            self.conv1 = conv3x3(in_channels, out_channels, kernel_size, self.stride)
            self.conv2 = conv3x3(out_channels, out_channels, kernel_size)
        else:
            self.conv1 = dilated_conv(in_channels, out_channels, kernel_size, dilation=3)
            self.conv2 = dilated_conv(out_channels, out_channels, kernel_size, dilation=3)

        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = BatchNorm(out_channels)

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))

        out2 += residual
        out = self.relu(out2)
        return out


class ResnetStraight(nn.Module):
    def __init__(
        self,
        num_filter,
        map_num,
        BatchNorm,
        block_nums=[3, 4, 6, 3],
        block=ResidualBlockWithDilation,
        kernel_size=5,
        stride=[1, 1, 2, 2],
    ):
        super(ResnetStraight, self).__init__()
        self.in_channels = num_filter * map_num[0]
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        self.block_nums = block_nums
        self.kernel_size = kernel_size

        self.layer1 = self.blocklayer(
            block,
            num_filter * map_num[0],
            self.block_nums[0],
            BatchNorm,
            kernel_size=self.kernel_size,
            stride=self.stride[0],
        )
        self.layer2 = self.blocklayer(
            block,
            num_filter * map_num[1],
            self.block_nums[1],
            BatchNorm,
            kernel_size=self.kernel_size,
            stride=self.stride[1],
        )
        self.layer3 = self.blocklayer(
            block,
            num_filter * map_num[2],
            self.block_nums[2],
            BatchNorm,
            kernel_size=self.kernel_size,
            stride=self.stride[2],
        )

    def blocklayer(self, block, out_channels, block_nums, BatchNorm, kernel_size, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(
                    self.in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                BatchNorm(out_channels),
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                BatchNorm,
                kernel_size,
                stride,
                downsample,
                is_top=True,
            )
        )
        self.in_channels = out_channels
        for i in range(1, block_nums):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    BatchNorm,
                    kernel_size,
                    is_activation=True,
                    is_top=False,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return out3


class UVDocnet(nn.Module):
    def __init__(self, num_filter, kernel_size=5):
        super(UVDocnet, self).__init__()
        self.num_filter = num_filter
        self.in_channels = 3
        self.kernel_size = kernel_size
        self.stride = [1, 2, 2, 2]

        BatchNorm = nn.BatchNorm2d
        act_fn = nn.ReLU(inplace=True)
        map_num = [1, 2, 4, 8, 16]

        self.resnet_head = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.num_filter * map_num[0],
                bias=False,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.kernel_size // 2,
            ),
            BatchNorm(self.num_filter * map_num[0]),
            act_fn,
            nn.Conv2d(
                self.num_filter * map_num[0],
                self.num_filter * map_num[0],
                bias=False,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.kernel_size // 2,
            ),
            BatchNorm(self.num_filter * map_num[0]),
            act_fn,
        )

        self.resnet_down = ResnetStraight(
            self.num_filter,
            map_num,
            BatchNorm,
            block_nums=[3, 4, 6, 3],
            block=ResidualBlockWithDilation,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )

        map_num_i = 2
        self.bridge_1 = nn.Sequential(
            dilated_conv_bn_act(
                self.num_filter * map_num[map_num_i],
                self.num_filter * map_num[map_num_i],
                act_fn,
                BatchNorm,
                dilation=1,
            )
        )

        self.bridge_2 = nn.Sequential(
            dilated_conv_bn_act(
                self.num_filter * map_num[map_num_i],
                self.num_filter * map_num[map_num_i],
                act_fn,
                BatchNorm,
                dilation=2,
            )
        )

        self.bridge_3 = nn.Sequential(
            dilated_conv_bn_act(
                self.num_filter * map_num[map_num_i],
                self.num_filter * map_num[map_num_i],
                act_fn,
                BatchNorm,
                dilation=5,
            )
        )

        self.bridge_4 = nn.Sequential(
            *[
                dilated_conv_bn_act(
                    self.num_filter * map_num[map_num_i],
                    self.num_filter * map_num[map_num_i],
                    act_fn,
                    BatchNorm,
                    dilation=d,
                )
                for d in [8, 3, 2]
            ]
        )

        self.bridge_5 = nn.Sequential(
            *[
                dilated_conv_bn_act(
                    self.num_filter * map_num[map_num_i],
                    self.num_filter * map_num[map_num_i],
                    act_fn,
                    BatchNorm,
                    dilation=d,
                )
                for d in [12, 7, 4]
            ]
        )

        self.bridge_6 = nn.Sequential(
            *[
                dilated_conv_bn_act(
                    self.num_filter * map_num[map_num_i],
                    self.num_filter * map_num[map_num_i],
                    act_fn,
                    BatchNorm,
                    dilation=d,
                )
                for d in [18, 12, 6]
            ]
        )

        self.bridge_concat = nn.Sequential(
            nn.Conv2d(
                self.num_filter * map_num[map_num_i] * 6,
                self.num_filter * map_num[2],
                bias=False,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BatchNorm(self.num_filter * map_num[2]),
            act_fn,
        )

        self.out_point_positions2D = nn.Sequential(
            nn.Conv2d(
                self.num_filter * map_num[2],
                self.num_filter * map_num[0],
                bias=False,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
            ),
            BatchNorm(self.num_filter * map_num[0]),
            nn.PReLU(),
            nn.Conv2d(
                self.num_filter * map_num[0],
                2,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
            ),
        )

        self.out_point_positions3D = nn.Sequential(
            nn.Conv2d(
                self.num_filter * map_num[2],
                self.num_filter * map_num[0],
                bias=False,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
            ),
            BatchNorm(self.num_filter * map_num[0]),
            nn.PReLU(),
            nn.Conv2d(
                self.num_filter * map_num[0],
                3,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.2)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                nn.init.xavier_normal_(m.weight, gain=0.2)

    def forward(self, x):
        resnet_head = self.resnet_head(x)
        resnet_down = self.resnet_down(resnet_head)
        bridge_1 = self.bridge_1(resnet_down)
        bridge_2 = self.bridge_2(resnet_down)
        bridge_3 = self.bridge_3(resnet_down)
        bridge_4 = self.bridge_4(resnet_down)
        bridge_5 = self.bridge_5(resnet_down)
        bridge_6 = self.bridge_6(resnet_down)
        bridge_concat = torch.cat([bridge_1, bridge_2, bridge_3, bridge_4, bridge_5, bridge_6], dim=1)
        bridge = self.bridge_concat(bridge_concat)

        out_point_positions2D = self.out_point_positions2D(bridge)
        out_point_positions3D = self.out_point_positions3D(bridge)

        return out_point_positions2D, out_point_positions3D

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


if __name__ == "__main__":
    main()