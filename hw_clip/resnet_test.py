import torch
# import torchvision.models as models
# import clip
from model import ModifiedResNet

model = ModifiedResNet(layers=50, output_dim=1024, heads=3)

# 다운로드한 가중치 파일의 경로를 설정합니다.
weight_path = './RN50x64.pt'

# 가중치를 로드합니다. torch.jit.load를 사용합니다.
model = torch.jit.load(weight_path)

# 모델을 평가 모드로 전환합니다.
model.eval()

# 이제 모델을 사용할 준비가 완료되었습니다.
# 예를 들어, 입력 데이터를 생성하고 예측을 수행할 수 있습니다.
input_tensor = torch.randn(1, 3, 224, 224)  # 임의의 입력 데이터 (배치 크기 1, RGB 이미지 224x224)
output = model.encode_image(input_tensor)

print(output)