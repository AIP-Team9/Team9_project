import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

#CPU 혹은 GPU 사용, GPU우선적으로 사용가능
device = torch.device("cuda:0" if torch.cuda.is_avaliable() else "cpu")

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
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2) #overfiting 방지를 위한 drop
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    #현재는 미세먼지 농도 "수치"를 예측하는것인 모델이고
    #"수치"를 또 분류하려면 layer를 추가하면됨 ex)softmax(다중분류  ex:저 중 고) , binary(이진분류 ex:저 고)


model = MLP()

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

     #학습 결과를 출력, 불필요할우 생략가능
    if (epoch+1) % 10 == 0:  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') 

# 모델 평가는 다른파일에서 실행하므로 이 파일에서는 test후 결과를 nparray로 반환하기까지만함

# 예측 함수 정의
def predict(model, X_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
    return y_pred.numpy()

# 예측후 결과 저장
predicted_values = predict(model, X_test)

#결과를 npy값으로 저장, 다음파일에서 사용가능
np.save('MLP.npy', predicted_values)