#%%
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import os
import numpy as np
#%%
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len = 96, pred_len = 24):
        super().__init__()
        self.seq_len = seq_len # 과거 입력 시계열 길이
        self.pred_len = pred_len # 미래 예측 시계열 길이
        self.data = data # 전체 시계열 데이터
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1 # 슬라이딩 윈도우로 만들 수 있는 총 샘플 개수 
    
    def __getitem__(self, idx):
        x = self.data[idx : idx+self.seq_len] # 과거 시계열 데이터 = 모델 입력 
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, 0:1] # 미래 예측 구간 = target
        return  torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_etth1(csv_path, seq_len=96, pred_len=24): # csv 파일을 읽어옴
    df = pd.read_csv(csv_path) # CSV 읽어옴 
    df = df.drop(columns=["date"], errors="ignore")  # 날짜를 제거하고 넘파이 배열로 반환함.
    data = df.values
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return TimeSeriesDataset(data, seq_len, pred_len)

if __name__ == '__main__':
    dataset = load_etth1("/Users/optim/Desktop/Time_series/MP3Net/ett/ETTh1.csv")
    print("Sample: ", dataset[0][0].shape, dataset[0][1].shape)
    print(dataset)
    print(dataset[0])
    print(dataset[0][0]) # 96*7
    print(dataset[0][1]) # 24*1 
    print(len(dataset)) # 17301 = 17420 - 96 - 24 + 1 

#%%
etth1 = pd.read_csv("/Users/optim/Desktop/Time_series/MP3Net/ett/ETTh1.csv")
print(etth1)

dataset = load_etth1("/Users/optim/Desktop/Time_series/MP3Net/ett/ETTh1.csv")
print("dataset:", dataset)  # None이면 여기서 바로 알 수 있음

df = pd.DataFrame(dataset.data)
df.to_csv("saved_timeseries.csv", index=False)

