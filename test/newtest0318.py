
def perform_ocr(processed_image_path):
    try:
        # 使用传入的 processed_image_path 进行 OCR 操作
        with open(processed_image_path, 'rb') as image_file:
            image = Image.open(image_file)
            text = pytesseract.image_to_string(image)
            return text
    except Exception as e:
        print(f"OCR Error: {e}")
        return None


def test_ocr():
    processed_image_path = "E:\\PythonProject1\\UVDoc-main\\myfile\\visualize\\app_data/log/temp.png"
    result = perform_ocr(processed_image_path)
