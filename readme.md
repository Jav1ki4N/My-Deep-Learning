# My-Deep-Learning

YOLO use PyTorch model by default, but Rockchip SOC's NPU can only process `.rknn` files, that's why we want to use **RKNN-Toolkit2** to convert `.onnx` model.  

To install **RKNN-Toolkit2**, see [**here**](https://github.com/airockchip/rknn-toolkit2).

[**rknn-model-zoo**](https://github.com/airockchip/rknn_model_zoo) also provides some deployment examples as refs.

# My Environment

## x86 machine
- OS: Ubuntu 22.04.5 LTS x86_64 
- Host: ASUS TUF Gaming F15 FX507VV_FX507VV 1.0 
- Kernel: 6.8.0-40-generic 
- CPU: 13th Gen Intel i7-13700H (20) @ 4.800GHz 
- GPU: NVIDIA GeForce RTX 4060 Laptop

## Board
- OS: Ubuntu 22.04 
- Host: lubancat3 RK3576 8+64GB

                                                                   
                                                                   
