#%%
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import csv
import copy
import matplotlib.pyplot as plt
import pandas as pd

from model import MP3Net
from utils import hybrid_loss # 손실함수 계산 
from dataset import load_etth1 
from logger import init_metric_log, append_metric_log
from train_utils import train_one_epoch, evaluate
#%%
# ====== 하이퍼파라미터 ======
SEQ_LEN = 336 # 예측에 사용할 과거 시계열의 길이 = lookback window 
PRED_LEN = 96 # 예측 할 시계열의 길이 
INPUT_DIM = 7 # 변수의 개수 
BATCH_SIZE = [16,32,64,128]
EPOCHS = 10
LR = 0.001
PATCH_SIZES = [8, 32] # patch scale = 2 
STRIDE_SIZES = [4, 16] 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_evaluate(batch_size, epochs=10, save_prefix=""): # 주어진 배치 사이즈로 전체 학습 수행
    print(f"\ Training with batch_size={batch_size}...\n")

    # ====== 데이터 로딩 ======
    dataset = load_etth1("/Users/optim/Desktop/Time_series/MP3Net/ett/ETTh1.csv", seq_len=SEQ_LEN, pred_len=PRED_LEN)
    # data split 단계 : 0.6 : 0.2 : 0.2 
    total_len = len(dataset) # 전체 데이터셋의 길이
    train_len = int(0.6 * total_len) # train 데이터셋의 길이
    val_len = int(0.2 * total_len) 
    test_len = total_len - train_len - val_len
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)  # reproducibility
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ===== 모델 정의 =====
    model = MP3Net(
        patch_sizes = PATCH_SIZES, # [8, 32]
        stride_sizes = STRIDE_SIZES, # [4, 16] 
        input_dim = INPUT_DIM, # 7 
        output_dim=1, 
        pred_len=PRED_LEN 
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # metric 저장용 
    # 로그 초기화
    log_path = f"{save_prefix}metrics_bs{batch_size}.csv" # 배치 사이즈별 결과 저장
    init_metric_log(log_path)


    best_val_loss = float('inf') # 가장 낮은 검증 손실
    best_model_state = None # 가장 좋은 모델 파라미터 저장 공간 
    wait = 0 # early stopping 기다림 카운터 
    patience = 10 # patience만큼 개선 없으면 종료하기

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_mse', 'train_mae',
                        'val_loss', 'val_mse', 'val_mae'])


    for epoch in range(1, EPOCHS + 1): # 총 epoch만큼 반복하기
        train_loss, train_mse, train_mae = train_one_epoch(model, train_loader, optimizer, hybrid_loss, DEVICE) # 학습하기
        val_loss, val_mse, val_mae = evaluate(model, val_loader, hybrid_loss, DEVICE) # 성능평가하기
        append_metric_log(log_path, epoch, train_loss, train_mse,train_mae, val_loss, val_mse, val_mae)
        
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss # best_val_loss 업데이트
            best_model_state = copy.deepcopy(model.state_dict()) # model.state_dict()를 복사해서 저장하기
            torch.save(best_model_state, f"{save_prefix}best_model_bs{batch_size}.pt") # 배치 사이즈별 최적 모델 저장
            print(f" Best model saved at epoch {epoch}")
            wait = 0 # 성능이 좋아졌으니 기다릴 필요 없음. 
        else: # patience만큼 기다려도 성능이 개선되지 않으면, 학습 중단하기 early stopping
            wait += 1
            if wait >= patience:
                print(f" Early stopping at epoch {epoch}")
                break

    # 학습 종료 후 test 평가
    model.load_state_dict(torch.load(f"{save_prefix}best_model_bs{batch_size}.pt")) # 이전에 저장해둔 best model의 파라미터들을 불러옴. model = 성능이 가장 좋은 모델로 설정됨.
    test_loss, test_mse, test_mae = evaluate(model, test_loader, hybrid_loss, DEVICE) # 불러온 best model로 test set에 대해 예측하고 평가하기
    print(f"[TEST] Loss: {test_loss:.4f} | MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")
    return best_val_loss

# ====== Grid Search ======
if __name__ == "__main__":
    mode = "grid"  # "grid" 또는 "final" 로 바꿔서 실행할 수 있음

    if mode == "grid": # 각 batch size에 대해 10에폭씩 학습하고 val loss 기준으로 best batch size tjsxor 
        print("\n[Grid Search 시작]")
        candidate_batch_sizes = [16, 32, 64, 128]
        best_bs = None
        best_loss = float('inf')

        for bs in candidate_batch_sizes:
            val_loss = train_and_evaluate(bs, epochs=10, save_prefix="grid_")
            if val_loss < best_loss:
                best_loss = val_loss
                best_bs = bs

        print(f"\n[✔] Best batch size: {best_bs} (Val Loss: {best_loss:.4f})")
        with open("best_batch_size.txt", "w") as f:
            f.write(str(best_bs))

    elif mode == "final":
        print("\n[최종 학습 시작]")
        with open("best_batch_size.txt", "r") as f:
            best_bs = int(f.read())
        train_and_evaluate(best_bs, epochs=100, save_prefix="final_")

    else:
        raise ValueError("mode must be 'grid' or 'final'")



# %%
