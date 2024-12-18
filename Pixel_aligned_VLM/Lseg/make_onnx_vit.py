import torch
from modules.lseg_module import LSegNet

labels = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool', 'pillow', 'screen', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen', 'computer', 'swivel', 'boat', 'bar', 'arcade', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television', 'airplane', 'dirt', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer', 'canopy', 'washer', 'plaything', 'swimming', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic', 'tray', 'ashcan', 'fan', 'pier', 'crt', 'plate', 'monitor', 'bulletin', 'shower', 'radiator', 'glass', 'clock', 'flag']

# Path to the model weights
weight_path = '/home/jihoon-epitone/Downloads/Pixel_aligned_VLM/Lseg/checkpoints/demo_e200.ckpt'

# Initialize the model with the specified configurations
model = LSegNet(
    labels=labels,
    path=weight_path,
    backbone='clip_vitl16_384',
    features=256,
    crop_size=384,
    arch_option=0,
    block_depth=0,
    activation='relu',
)

# Load pre-trained weights
model.load(path=weight_path)
model.eval()  # Set the model to evaluation mode

# Move the model and input to the GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device).half()  # Ensure the model is moved to the correct device

dummy_input = torch.randn(1, 3, 384, 384).to(device).half()  # Define a dummy input tensor and move it to GPU

# Ensure all model parameters are moved to the GPU
for param in model.parameters():
    param.data = param.data.to(device).half()

# Check that all parameters are on the correct device
for name, param in model.named_parameters():
    if str(param.device) != 'cuda:0':
        print(f"{name} is on {param.device}")

output = model(dummy_input)

# Export the model to ONNX format
with torch.no_grad():  # Disable gradient computation
    torch.onnx.export(
        model,
        dummy_input,
        "lseg_model_vit.onnx",  # Name of the exported ONNX file
        export_params=True,  # Export model parameters
        opset_version=13,  # Specify the ONNX opset version
        do_constant_folding=True,  # Enable constant folding for optimization
        input_names=['input'],  # Name of the input node
        output_names=['output'],  # Name of the output node
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Support dynamic batch sizes
        , verbose=True
    )

# Release unused GPU memory
torch.cuda.empty_cache()
