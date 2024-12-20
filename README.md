# pytorch_trt_comparison

## Requirements
Make sure you first follow the instructions here (Lseg section): https://github.com/Lab-of-AI-and-Robotics/Pixel_aligned_VLM

- Make sure you have downloaded "demo_e200.ckpt" file.

## Pytorch models

### CLIP
1. To make onnx file
```
python hw_clip/CLIP_resnet_test.py --make_onnx_file
```
2. To measure execution time
```
python hw_clip/CLIP_resnet_test.py --measure_time
```
3. To do both
```
python hw_clip/CLIP_resnet_test.py --make_onnx_file --measure_time
```
4. To load model only
```
python hw_clip/CLIP_resnet_test.py
```

Execution time: 4.363ms


### Lseg (Resnet base)
1. To make onnx file
```
python Pixel_aligned_VLM/Lseg/Lseg_resnet_test.py
```
2. To measure execution time
```
python Pixel_aligned_VLM/Lseg/Lseg_resnet_test.py --measure_time
```
3. To do both
```
python Pixel_aligned_VLM/Lseg/Lseg_resnet_test.py --make_onnx_file --measure_time
```
4. To load model only
```
python Pixel_aligned_VLM/Lseg/Lseg_resnet_test.py
```

Execution time: 22.162ms

## Onnx to TRT conversion

## Trt models
CLIP: 1.931ms
Lseg: 15.457ms