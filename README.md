# pytorch_trt_comparison

## CLIP

pytorch model execution time command 
```
python hw_clip/resnet_test.py
```

Pytorch: 4.363ms
Trt: 1.931ms

## Lseg (Resnet base)

pytorch model execution time command 
```
python Pixel_aligned_VLM/Lseg/make_onnx_resnet.py
```

Pytorch: 22.162ms
Trt: 15.457ms