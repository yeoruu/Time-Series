#%%
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import MP3Net
from utils import hybrid_loss # 손실함수 계산 
from dataset import load_etth1 
#%%
# ====== 하이퍼파라미터 ======
SEQ_LEN = 96 # 예측에 사용할 과거 시계열의 길이 
PRED_LEN = 24 # 예측 할 시계열의 길이
INPUT_DIM = 7 # 변수의 개수 
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
PATCH_SIZES = [16, 32] # patch scale = 2 
STRIDE_SIZES = [8, 16] 

# ====== 데이터 로딩 ======
dataset = load_etth1("/Users/optim/Desktop/Time_series/MP3Net/ett/ETTh1.csv", seq_len=SEQ_LEN, pred_len=PRED_LEN)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===== 모델 정의 =====
model = MP3Net(
    patch_sizes = PATCH_SIZES,
    stride_sizes = STRIDE_SIZES,
    input_dim = INPUT_DIM,
    output_dim=1,
    pred_len=PRED_LEN 
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ====== 옵티마이저 ======
optimizer = optim.Adam(model.parameters(), lr=LR)
#%%
# ====== 학습 루프 ======
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    total_mse = 0
    total_mae = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, (batch_x, batch_y) in enumerate(loop):
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)

        optimizer.zero_grad()
        preds = model(batch_x)

        loss_value, mse_value, mae_value = hybrid_loss(preds, batch_y)
        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.item()
        total_mse += mse_value.item()
        total_mae += mae_value.item()

        loop.set_postfix(loss=loss_value.item(), mse=mse_value.item(), mae=mae_value.item())

    avg_loss = total_loss / len(train_loader)
    avg_mse = total_mse / len(train_loader)
    avg_mae = total_mae / len(train_loader)

    print(f"[EPOCH {epoch}] Train Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | MAE: {avg_mae:.4f}")
# %%

