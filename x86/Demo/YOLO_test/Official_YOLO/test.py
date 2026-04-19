# test.py
# This script quick test YOLO in x86 machine
# I'm using a Ultralytics official YOLO11n model
# i4N@2026
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("https://ultralytics.com/images/zidane.jpg")
results[0].save(filename="result.jpg")