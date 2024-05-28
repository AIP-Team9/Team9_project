import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# 데이터 불러오기 (전처리 완료된 데이터라고 가정)
data = pd.read_csv('data.csv')

# 특성과 타겟 변수 분리
X = data.drop(columns=[' /타켓 컬럼 명/ ']).values
y = data['/타켓 컬럼 명/'].values

# Tensor로 변환
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 데이터셋 나누기 (훈련 세트 : 테스트 세트 = 8 : 2 사이즈로)
dataset_size = len(X)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
X_train, X_test = torch.split(X, [train_size, test_size])
y_train, y_test = torch.split(y, [train_size, test_size])

# DataLoader 정의
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.input_size=input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_putsize=output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) #일단은 nn.LSTM을 사용 추후에 커스텀을 할 예정
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        output, _ = self.lstm(x, (h0, c0))
        output = self.fc(output[:, -1, :])
        return output

# 모델 초기화

input_size = 2
output_size = 1
hidden_size=64
num_layers=2
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 손실 함수 및 옵티마이저 정의
learning_rate=0.001
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
num_epchos=10 #알맞는 값으로 변경 임시 epoch 10
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 예측 함수 정의
def predict(model, X):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
    return y_pred.numpy()

# 예측 결과 저장
predicted_values = predict(model, X_test)

np.save('LSTM.npy', predicted_values)