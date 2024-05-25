import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.opotim as optim
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# 데이터 로드
train_data = pd.read_csv("train_data.csv")
#val_data = pd.read_csv("val_data.csv")
test_data = pd.read_csv("test_data.csv")

# 특성과 타겟 변수 분리
X_train = train_data.drop(columns=["관측미세먼지"]).values
X_test = test_data.drop(columns=["관측미세먼지"]).values
y_train = train_data["관측미세먼지"].values
y_test = test_data["관측미세먼지"].values

# 시계열 자료로 사용하기위해 관측시간을 활용가능한 값으로 변경
train_data['관측시간'] = pd.to_datetime(data['관측시간'])
train_data.set_index('관측시간', inplace=True)
test_data['관측시간'] = pd.to_datetime(data['관측시간'])
test_data.set_index('관측시간', inplace=True)

# 관측시간을 제대로 활용하기 위해 관측지점 라벨 인코딩
label_encoder = LabelEncoder()
train_data['관측지점'] = label_encoder.fit_transform(train_data['관측지점'])
test_data['관측지점'] = label_encoder.transform(test_data['관측지점'])

# 각각 train과 test를 알맞게 데이터와 매치, 추후 필요없는 특성이면 drop, 아니라면 다른 전처리 필요
X_train = train_data.drop(columns=["관측미세먼지"]).apply(pd.to_numeric, errors='coerce').fillna(0).values
X_test = test_data.drop(columns=["관측미세먼지"]).apply(pd.to_numeric, errors='coerce').fillna(0).values
y_train = train_data["관측미세먼지"].apply(pd.to_numeric, errors='coerce').fillna(0).values
y_test = test_data["관측미세먼지"].apply(pd.to_numeric, errors='coerce').fillna(0).values

# 데이터 정규화
scaler = MinMaxScaler()
X_train_scaled_data = scaler.fit_transform(X_train)
X_test_scaled_data=scaler.fit_transform(X_test)
# 시퀀스 생성
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length-1,-1])
    return np.array(X), np.array(y)

seq_length = 24 # 하루를 기준으로 잡음
X_train, y_train = create_sequences(X_train_scaled_data, seq_length)
X_test,y_test=create_sequences(X_test_scaled_data, seq_length)

# 데이터셋 축소
test_size = int(0.01 * len(X_train))
X_train = X_train[:test_size]
y_train = y_train[:test_size]

# 텐서로 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# DataLoader 정의
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# LSTM 모델 정의
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out

# 모델 하이퍼파라미터 설정
input_size = X_train.shape[2]
hidden_size = 64
num_layers = 5
output_size = 1

# 모델, 손실 함수, 옵티마이저 정의
model = LSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델 학습
num_epochs = 5
for epoch in range(num_epochs):
    running_loss=0.0
    for X_batch, y_batch in train_loader:
        # 데이터를 GPU로 이동
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")



# 예측 함수 정의
def predict(model, X_test):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_outputs = test_outputs.squeeze().cpu().numpy()
    return test_outputs

# 예측후 결과 저장
predicted_values = predict(model, X_test)

#결과를 npy값으로 저장, 다음파일에서 사용가능
np.save('LSTM.npy', predicted_values)
