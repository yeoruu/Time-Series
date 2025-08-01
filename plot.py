
import matplotlib.pyplot as plt
import torch
@torch.no_grad()
def plot_predictions(model, dataloader, device, n_batches=1):
    model.eval()
    preds = []
    trues = []

    for i, (batch_x, batch_y) in enumerate(dataloader):
        if i >= n_batches:  # n_batches만큼만 시각화
            break

        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        pred = model(batch_x)
        preds.append(pred.cpu().numpy())
        trues.append(batch_y.cpu().numpy())

    import numpy as np
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # 첫 시계열만 그려보기
    plt.figure(figsize=(10, 4))
    plt.plot(trues[0].flatten(), label='True')
    plt.plot(preds[0].flatten(), label='Predicted')
    plt.legend()
    plt.title("Test Prediction vs Ground Truth")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_prediction_plot.png")  # 저장도 가능
    plt.show()

plot_predictions(model, test_loader, device, n_batches=1)
#%% epoch별로 metric 그리는 그래프
import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기
df = pd.read_csv("metrics.csv")

# 그래프 크기 설정
plt.figure(figsize=(15, 4))

# ----- 1. Loss -----
plt.subplot(1, 3, 1)
plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.grid(True)

# ----- 2. MSE -----
plt.subplot(1, 3, 2)
plt.plot(df['epoch'], df['train_mse'], label='Train MSE')
plt.plot(df['epoch'], df['val_mse'], label='Val MSE')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE per Epoch")
plt.legend()
plt.grid(True)

# ----- 3. MAE -----
plt.subplot(1, 3, 3)
plt.plot(df['epoch'], df['train_mae'], label='Train MAE')
plt.plot(df['epoch'], df['val_mae'], label='Val MAE')
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("MAE per Epoch")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("epoch_metrics_plot.png")  # 저장 (선택)
plt.show()
