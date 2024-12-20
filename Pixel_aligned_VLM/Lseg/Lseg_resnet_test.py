import os
import argparse
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from glob import glob
from encoding.models.sseg import BaseNet
from modules.models.lseg_net_zs import LSegRN_img_only
import onnx
import time
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from modules.models.lseg_blocks_zs import Interpolate

parser = argparse.ArgumentParser(description="Run visual component of Lseg with optional ONNX export.")
parser.add_argument("--make_onnx_file", action="store_true", help="Set to export the model to ONNX")
parser.add_argument("--measure_time", action="store_true", help="Set to measure inference time")
args = parser.parse_args()

# flags
make_onnx_file = args.make_onnx_file
measure_time = args.measure_time

labels = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool', 'pillow', 'screen', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen', 'computer', 'swivel', 'boat', 'bar', 'arcade', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television', 'airplane', 'dirt', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer', 'canopy', 'washer', 'plaything', 'swimming', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic', 'tray', 'ashcan', 'fan', 'pier', 'crt', 'plate', 'monitor', 'bulletin', 'shower', 'radiator', 'glass', 'clock', 'flag']

model = LSegRN_img_only(
    head = nn.Sequential(
        Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
    ),
    features=256,
    backbone="clip_resnet101",
    readout="project",
    channels_last=False,
    use_bn=False,
    arch_option=0
)

model.load(path='./Pixel_aligned_VLM/Lseg/checkpoints/demo_e200.ckpt')

# eval mode
model.eval()

print("Lseg pytorch model loaded")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
dummy_input = torch.randn(1, 3, 384, 384).to(device)

if measure_time:
    # time measurement
    num_iterations = 100  # 반복 횟수
    total_time = 0.0

    for _ in tqdm(range(num_iterations), desc="Inference Timing"):
        start = time.time()
        with torch.no_grad():  
            output = model(dummy_input)
        end = time.time()
        total_time += (end - start)

    average_time_per_inference = total_time / num_iterations
    print(f"Average time per inference: {average_time_per_inference:.6f} seconds")


if make_onnx_file:
    export_model = "./hw_clip/cpp_ver/engine_files/lseg_resnet.onnx"

    # export to onnx
    torch.onnx.export(
        model,
        dummy_input,
        export_model,
        export_params=True,
        opset_version=13,  # or higher
        do_constant_folding=True,  # Try turning this off if it's causing issues
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )