import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# y_test 가져오기
data = pd.read_csv('/content/drive/MyDrive/test_seoul.csv')

y = data['관측미세먼지'].values
y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

#dataset_size = len(y)
#train_size = int(0.8 * dataset_size)
#test_size = dataset_size
#y_train, y_test = torch.split(y, [train_size, test_size])

# 예측값 가져오기
actual_values = y.cpu().numpy()
predicted_values_MLP = np.load('MLP.npy')
#predicted_values_Transformer = np.load('Transformer.npy')
#predicted_values_LSTM = np.load('LSTM.npy')

def evaluate_model(actual_values, predicted_values):

    # RMSE 계산
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

    # R-squared 계산
    r2 = r2_score(actual_values, predicted_values)
    
    return rmse, r2

def visualize_model(actual_values, predicted_values, model_name):
    
    time = np.arange(len(predicted_values))
    
    plt.figure(figsize=(14, 7))
    plt.plot(time, actual_values, label='Actual', color='blue')
    plt.plot(time, predicted_values, label='Predicted', color='red')
    plt.xlabel('Time')
    plt.ylabel('????(μg/m³)')
    plt.title(f'{model_name} Actual vs Predicted')
    plt.legend()
    plt.show()


rmse_MLP, r2_MLP = evaluate_model(actual_values, predicted_values_MLP)
visualize_model(actual_values, predicted_values_MLP, 'MLP')
print(f'RMSE of MLP Model : {rmse_MLP:.4f}')
print(f'R-squared of MLP Model : {r2_MLP:.4f}')

#rmse_Transformer, r2_Transformer = evaluate_model(actual_values, predicted_values_Transformer)
#visualize_model(actual_values, predicted_values_Transformer, 'Transformer')
#print(f'RMSE of Transformer Model : {rmse_Transformer:.4f}')
#print(f'R-squared of Transformer Model : {r2_Transformer:.4f}')

#rmse_LSTM, r2_LSTM = evaluate_model(actual_values, predicted_values_LSTM)
#visualize_model(actual_values, predicted_values_LSTM, 'LSTM')
#print(f'RMSE of LSTM Model : {rmse_LSTM:.4f}')
#print(f'R-squared of LSTM Model : {r2_LSTM:.4f}')


