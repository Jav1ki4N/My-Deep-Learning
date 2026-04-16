from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("https://ultralytics.com/images/zidane.jpg")
results[0].save(filename="result.jpg")