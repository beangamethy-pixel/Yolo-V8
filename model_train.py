from ultralytics import YOLO

# 加载YOLOv8模型，选择一个预训练模型（比如yolov8n.pt，yolov8s.pt等）
model = YOLO('yolov8n.pt')

# 使用数据集和配置文件训练
model.train(
    data="yolo_v8_app_detection.yaml",
    epochs=10,
    batch=16,
    imgsz=640,
)

