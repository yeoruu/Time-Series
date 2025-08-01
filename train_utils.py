#%%
import torch
from tqdm import tqdm

#%%
# ====== 학습 루프 ======
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    print("[DEBUG] train_one_epoch 진입")
    model.train()
    total_loss = total_mse = total_mae = 0
    loop = tqdm(dataloader, desc="Training", leave=True)

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)

        optimizer.zero_grad()
        preds = model(batch_x)
        loss, mse, mae = criterion(preds, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_mae += mae.item()

        loop.set_postfix(loss=loss.item(), mse=mse.item(), mae=mae.item())

    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_mae = total_mae / len(dataloader)
    return avg_loss, avg_mse, avg_mae
# %%
# ====== 평가 루프 ======
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_mse, total_mae = 0, 0, 0

    loop = tqdm(dataloader, desc="Evaluating", leave=True)


    for batch_x, batch_y in loop:
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
        preds = model(batch_x)

        loss, mse, mae = criterion(preds, batch_y)
        total_loss += loss.item()
        total_mse += mse.item()
        total_mae += mae.item()

        loop.set_postfix(loss=loss.item(), mse=mse.item(), mae=mae.item())

    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_mae = total_mae / len(dataloader)
    
    return avg_loss, avg_mse, avg_mae 

# %%
