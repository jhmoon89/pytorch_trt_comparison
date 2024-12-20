# pytorch_trt_comparison

## Requirements
Make sure you first follow the instructions here (Lseg section): https://github.com/Lab-of-AI-and-Robotics/Pixel_aligned_VLM

- Make sure you have downloaded "demo_e200.ckpt" file.

## CLIP

1. pytorch model

a. To make onnx file
```
python hw_clip/CLIP_resnet_test.py --make_onnx_file
```
b. To measure execution time
```
python hw_clip/CLIP_resnet_test.py --measure_time
```
c. To do both
```
python hw_clip/CLIP_resnet_test.py --make_onnx_file --measure_time
```
d. To load model only
```
python hw_clip/CLIP_resnet_test.py
```

2. trt model

Pytorch: 4.363ms
Trt: 1.931ms

## Lseg (Resnet base)

1. pytorch model

a. To make onnx file
```
python Pixel_aligned_VLM/Lseg/Lseg_resnet_test.py
```
b. To measure execution time
```
python Pixel_aligned_VLM/Lseg/Lseg_resnet_test.py --measure_time
```
c. To do both
```
python Pixel_aligned_VLM/Lseg/Lseg_resnet_test.py --make_onnx_file --measure_time
```
d. To load model only
```
python Pixel_aligned_VLM/Lseg/Lseg_resnet_test.py
```

Pytorch: 22.162ms
Trt: 15.457ms