from ultralytics import YOLO

model = YOLO("rtmlib\weights\yolo11n.pt")
model.export(
    format="onnx",
    imgsz=640,
    simplify=True,
    nms=True
)