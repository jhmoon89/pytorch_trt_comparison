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

1. Give permission to sh file
```
chmod +x hw_clip/cpp_ver/engine_files/convert_onnx_to_trt.sh
```
2. Convert CLIP model from onnx to trt
```
hw_clip/cpp_ver/engine_files/convert_onnx_to_trt.sh hw_clip/cpp_ver/engine_files/clip_visual_component.onnx hw_clip/cpp_ver/engine_files/clip_visual_component.trt
```
3. Convert Lseg model from onnx to trt
```
hw_clip/cpp_ver/engine_files/convert_onnx_to_trt.sh hw_clip/cpp_ver/engine_files/lseg_resnet.onnx hw_clip/cpp_ver/engine_files/
lseg_resnet.trt
```

## Trt models

### CLIP
```
hw_clip/cpp_ver/build/./trt_model_test hw_clip/cpp_ver/engine_files/clip_visual_component.trt
```
Execution time: 1.931ms


### Lseg (Resnet base)
```
hw_clip/cpp_ver/build/./trt_model_test hw_clip/cpp_ver/engine_files/lseg_resnet.trt
```
Execution time: 15.457ms


## Execution time summary
| Model           | Pytorch Execution Time | ONNX to TRT Execution Time | TRT Execution Time |
|-----------------|------------------------|----------------------------|---------------------|
| **CLIP**        | 4.363 ms               | N/A                        | 1.931 ms            |
| **Lseg (Resnet)**| 22.162 ms             | N/A                        | 15.457 ms           |