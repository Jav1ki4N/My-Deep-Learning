# convert.py
# This script converts ONXX model to RKNN using RKNN-Toolkit in x86 machine
# Original reference: https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolo11/python/convert.py
# I'm using a Ultralytics official YOLO11n model in ONNX format
# i4N@2026

# Env
# - Python 3.8
# - RKNN-Toolkit 2.3.2

from rknn.api import RKNN

ONNX_PATH = './yolo11n.onnx'

# Create RKNN instance
rknn = RKNN(verbose=False)

# RKNN model config
# mean_values & std_values: preprocessing parameters for normalization
# platform:                 the target rockchip platform, e.g. 'rk3576'
# For details, see convert_ori.py
print("Configuring model.")
rknn.config( mean_values     = [[0,0,0]],
             std_values      = [[255,255,255]],
             target_platform = 'rk3576')
print("done")

# Load ONNX model
# ret is the return code: 0 for success, non-0 for failure
print("Loading ONNX model.")
ret = rknn.load_onnx(model=ONNX_PATH)
if ret != 0:
    print("Failed to load ONNX model.")
    exit(ret)
print("done")

# Build RKNN model
print("Building RKNN model.")
ret =rknn.build(do_quantization=False)
if ret != 0:
    print("Failed to build RKNN model.")
    exit(ret)
print("done")

# Export RKNN model
print("Exporting RKNN model.")
ret = rknn.export_rknn('./yolo11n.rknn')
if ret != 0:
    print("Failed to export RKNN model.")
    exit(ret)
print("done")

rknn.release()

