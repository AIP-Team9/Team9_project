import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# CPU 혹은 GPU 사용, GPU 우선적으로 사용 가능
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 데이터 불러오기 (전처리 완료된 데이터라고 가정)
data = pd.read_csv('data.csv')

# 특성과 타겟 변수 분리
X = data.drop(columns=['/타켓 컬럼 명/']).values
y = data['/타켓 컬럼 명/'].values

# Tensor로 변환
X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

# 데이터셋 나누기 (훈련 세트: 테스트 세트 = 8:2 사이즈로)
dataset_size = len(X)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
X_train, X_test = torch.split(X, [train_size, test_size])
y_train, y_test = torch.split(y, [train_size, test_size])

# DataLoader 정의
train_dataset = TensorDataset(X_train, y_train)
# batch_size: 각 배치의 데이터 수
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 트랜스포머 모델 정의
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, model_dim))
    
    def forward(self, src):
        src = self.embedding(src) + self.pos_encoder
        src = src.permute(1, 0, 2)  # (batch_size, seq_len, model_dim) -> (seq_len, batch_size, model_dim)
        transformer_output = self.transformer(src, src)
        output = self.fc_out(transformer_output[-1])  # (seq_len, batch_size, model_dim) -> (batch_size, model_dim)
        return output

# 모델 초기화
input_dim = X.shape[1] # 입력 데이터의 차원
model_dim = 64 # 트랜스포머 모델의 차원. 32, 64, 128
num_heads = 4 # 멀티헤드 어텐션의 헤드 수. 2, 4, 8
num_layers = 2 # 인코더와 디코더 레이어의 수. 2, 4, 6
output_dim = 1 # 출력 차원 (예측하려는 값의 수)
model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)

# 손실 함수, 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # 학습 결과 출력, 불필요할 경우 생략 가능
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 모델 평가
def predict(model, X_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
    return y_pred.cpu().numpy()

# 예측 후 결과 저장
predicted_values = predict(model, X_test)

# 결과를 npy값으로 저장, 다음 파일에서 사용 가능
np.save('Transformer.npy', predicted_values)
