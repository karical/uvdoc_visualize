import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import sys

import cv2

import streamlit as st

import os
os.environ["STREAMLIT_SERVER_ENABLE_WATCH_DOG"] = "false"
ROOT = os.path.dirname(os.path.abspath(__file__))

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


from app_data.dependencies.utils import IMG_SIZE, bilinear_unwarping, load_model
from app_data.dependencies.streamlit_utils import load_yolo_model, unwarp_img, plot_results, perform_ocr


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


if __name__ == "__main__":
    main()