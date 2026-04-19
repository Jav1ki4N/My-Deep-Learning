# export.py
# This script converts pt model to onnx
# I'm using a Ultralytics official YOLO11n model
# i4N@2026

from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.export(format ="onnx",
             imgsz = [640,640],
             opset = 12,
             simplify = True,
)