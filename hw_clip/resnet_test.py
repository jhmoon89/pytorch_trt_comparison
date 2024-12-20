import torch
# import torchvision.models as models
from clip import load
# from model import ModifiedResNet

import time
import numpy as np
from tqdm import tqdm


# flags
make_onnx_file = False
measure_time = True

# Load the entire CLIP model
# model, _ = load("ViT-B/32")  # Or whichever variant you want
model, _ = load("RN50")  # Or whichever variant you want

# print(model)

# Extract the visual component
visual_component = model.visual

# # Save the weights of the visual component
# torch.save(visual_component.state_dict(), 'visual_weights.pth')

# Later, to load the visual weights back into the model
weight_path = './hw_clip/cpp_ver/engine_files/visual_weights.pth'
visual_component.load_state_dict(torch.load(weight_path))

if measure_time:
    iter_num = 1000
    time_list = np.zeros([iter_num])

    with torch.no_grad(): 
        for i in tqdm(range(iter_num)):
            input_tensor = torch.randn(1, 3, 224, 224).cuda().half()
            start_time = time.time()
            output = visual_component(input_tensor)
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_list[i] = elapsed_time
            # print(f"Execution time: {elapsed_time:.5f} seconds")

    print(f"Average time per inference: {np.mean(time_list):.6f} seconds")
    # print(np.mean(time_list))


# for _ in tqdm(range(num_iterations), desc="Inference Timing"):
#     start = time.time()
#     with torch.no_grad():  # 평가 모드에서 불필요한 그래디언트 계산 비활성화
#         output = model(dummy_input)
#     end = time.time()
#     total_time += (end - start)

# average_time_per_inference = total_time / num_iterations
# print(f"Average time per inference: {average_time_per_inference:.6f} seconds")

if make_onnx_file:
    # Define a dummy input for ONNX export (assuming 3x224x224 input for the visual component)
    dummy_input = torch.randn(1, 3, 224, 224).to(next(visual_component.parameters()).device)

    # Export the visual component to ONNX
    torch.onnx.export(
        visual_component,                    # Model to export
        dummy_input,                         # Dummy input
        "visual_component.onnx",             # Output ONNX file name
        export_params=True,                  # Store parameters in the model file
        opset_version=10,                    # ONNX opset version (adjust as needed)
        do_constant_folding=True,            # Simplify the model by folding constants
        input_names=['input'],               # Specify input name(s)
        output_names=['output'],             # Specify output name(s)
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Make batch dimension dynamic
    )

    print("Model has been successfully exported to visual_component.onnx")

##################################################################################
# model = ModifiedResNet(layers=50, output_dim=1024, heads=3)

# # 다운로드한 가중치 파일의 경로를 설정합니다.
# weight_path = './RN50x64.pt'

# # 가중치를 로드합니다. torch.jit.load를 사용합니다.
# model = torch.jit.load(weight_path)

# # 모델을 평가 모드로 전환합니다.
# model.eval()

# # 이제 모델을 사용할 준비가 완료되었습니다.
# # 예를 들어, 입력 데이터를 생성하고 예측을 수행할 수 있습니다.
# input_tensor = torch.randn(1, 3, 224, 224)  # 임의의 입력 데이터 (배치 크기 1, RGB 이미지 224x224)


# output = model.encode_image(input_tensor)

# print(output)