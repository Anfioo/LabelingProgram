import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QFileDialog, QSlider,
                             QMessageBox, QTextEdit, QGroupBox)
from PyQt5.QtCore import Qt
import json
import os

import os
import cv2
from ultralytics import YOLO

from tqdm import tqdm
from SvgRenderer import get_use_model_svg_icon, set_svg_icon_from_string


def detect_and_label(input_dir, output_label_dir, model_path, class_mapping, conf_threshold=0.8):
    """
    人脸检测并生成YOLO格式标签

    :param input_dir: 包含所有图片的输入目录
    :param output_label_dir: 标签输出目录
    :param model_path: 人脸检测模型路径
    :param class_mapping: 文件名到类别ID的映射字典
    :param conf_threshold: 人脸检测置信度阈值 (默认0.8)
    """
    # 加载人脸检测模型
    model = YOLO(model_path)

    # 创建输出目录
    os.makedirs(output_label_dir, exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in tqdm(image_files, desc='Processing images'):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # 从文件名解析类别
        class_id = None
        for prefix in class_mapping:
            if prefix.lower() in img_file.lower():
                class_id = class_mapping[prefix]
                break

        if class_id is None:
            continue

        # 人脸检测（使用指定置信度阈值）
        results = model.predict(img, imgsz=640, conf=conf_threshold)

        # 生成YOLO格式标签
        label_path = os.path.join(output_label_dir, os.path.splitext(img_file)[0] + '.txt')
        with open(label_path, 'w') as f:
            for box in results[0].boxes:
                if box.conf < conf_threshold:
                    continue

                # 转换为YOLO格式（归一化中心坐标和宽高）
                x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2 / w
                y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2 / h
                box_width = (box.xyxy[0][2] - box.xyxy[0][0]) / w
                box_height = (box.xyxy[0][3] - box.xyxy[0][1]) / h

                # 写入文件：class_id x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")


class FaceLabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自动标注工具")
        self.setGeometry(100, 100, 600, 600)

        # Initialize variables
        self.input_dir = ""
        self.output_dir = ""
        self.model_path = "../yolov8n-face-lindevs.pt"
        self.conf_threshold = 0.75
        self.class_mapping = {}

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel("自动标注工具")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Input directory
        input_group = QGroupBox("图片目录")
        input_layout = QHBoxLayout()
        self.input_edit = QLineEdit()
        input_btn = QPushButton("浏览...")
        input_btn.clicked.connect(self.browse_input_dir)
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(input_btn)
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        # Output directory
        output_group = QGroupBox("标签输出目录 (可选)")
        output_layout = QHBoxLayout()
        self.output_edit = QLineEdit()
        output_btn = QPushButton("浏览...")
        output_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_btn)
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)

        # Model path
        model_group = QGroupBox("模型路径")
        model_layout = QHBoxLayout()
        self.model_edit = QLineEdit(self.model_path)
        model_btn = QPushButton("浏览...")
        model_btn.clicked.connect(self.browse_model_path)
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(model_btn)
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # Confidence threshold
        conf_group = QGroupBox("置信度阈值")
        conf_layout = QVBoxLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 100)
        self.conf_slider.setValue(int(self.conf_threshold * 100))
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        self.conf_label = QLabel(f"{self.conf_threshold:.2f}")
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        conf_group.setLayout(conf_layout)
        main_layout.addWidget(conf_group)

        # Class mapping file
        mapping_group = QGroupBox("类别映射文件")
        mapping_layout = QVBoxLayout()

        file_layout = QHBoxLayout()
        self.mapping_edit = QLineEdit()
        mapping_btn = QPushButton("浏览...")
        mapping_btn.clicked.connect(self.browse_mapping_file)
        file_layout.addWidget(self.mapping_edit)
        file_layout.addWidget(mapping_btn)
        mapping_layout.addLayout(file_layout)

        # Preview area
        preview_label = QLabel("类别映射预览:")
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText("未加载类别映射文件")

        update_btn = QPushButton("更新预览")
        update_btn.clicked.connect(self.update_mapping_preview)

        mapping_layout.addWidget(preview_label)
        mapping_layout.addWidget(self.preview_text)
        mapping_layout.addWidget(update_btn)
        mapping_group.setLayout(mapping_layout)
        main_layout.addWidget(mapping_group)

        # Run button
        run_btn = QPushButton("开始处理")
        run_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        run_btn.clicked.connect(self.run_processing)
        main_layout.addWidget(run_btn)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        set_svg_icon_from_string(self, get_use_model_svg_icon())

    def browse_input_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if dir_path:
            self.input_dir = dir_path
            self.input_edit.setText(dir_path)

            # Set default output dir
            if not self.output_dir:
                parent_dir = os.path.dirname(dir_path)
                self.output_dir = os.path.join(parent_dir, "labels")
                self.output_edit.setText(self.output_dir)

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择标签输出目录")
        if dir_path:
            self.output_dir = dir_path
            self.output_edit.setText(dir_path)

    def browse_model_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Model Files (*.pt)")
        if file_path:
            self.model_path = file_path
            self.model_edit.setText(file_path)

    def browse_mapping_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择类别映射文件", "", "TXT Files (*.txt)")
        if file_path:
            self.mapping_edit.setText(file_path)

    def update_conf_label(self, value):
        self.conf_threshold = value / 100
        self.conf_label.setText(f"{self.conf_threshold:.2f}")

    def update_mapping_preview(self):
        mapping_file = self.mapping_edit.text()
        self.preview_text.clear()

        if mapping_file and os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    self.class_mapping = json.load(f)
                    get = self.class_mapping.get('class_to_id')
                    self.preview_text.setPlainText(str(get))
            except Exception as e:
                self.preview_text.setPlainText(f"加载失败: {str(e)}")
        else:
            self.preview_text.setPlainText("未加载类别映射文件")

    def run_processing(self):
        # Validate inputs
        if not self.input_dir:
            QMessageBox.critical(self, "错误", "请选择图片目录")
            return

        # Set default output dir if not specified
        if not self.output_dir:
            self.output_dir = os.path.join(self.input_dir, "labels")
            self.output_edit.setText(self.output_dir)

        # Check class mapping
        if not self.class_mapping:
            reply = QMessageBox.question(
                self, "警告",
                "未加载有效的类别映射文件，是否继续?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # Here you would call your detect_and_label function
        # detect_and_label(...)

        # 执行处理
        get = self.class_mapping.get('class_to_id')
        print(get)
        detect_and_label(
            input_dir=self.input_dir,
            output_label_dir=self.output_dir,
            model_path=self.model_path,
            class_mapping=get,
            conf_threshold=self.conf_threshold
        )

        QMessageBox.information(self, "信息", "开始处理...")
        print("Processing parameters:")
        print(f"Input dir: {self.input_dir}")
        print(f"Output dir: {self.output_dir}")
        print(f"Model path: {self.model_path}")
        print(f"Confidence threshold: {self.conf_threshold}")
        print("Class mapping:", self.class_mapping)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceLabelingApp()
    window.show()
    sys.exit(app.exec_())
